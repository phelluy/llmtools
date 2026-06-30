#!/usr/bin/env python3
"""Convertit un PDF en Markdown en utilisant l'API UNISTRA Qwen.

Algorithme incrémental :
  1. Convertir chaque page du PDF en image JPEG (base64).
  2. Pour chaque page, envoyer l'image + le contexte des pages précédentes
     au LLM Qwen via l'endpoint /v1/chat/completions.
  3. Extraire le markdown entre <markdown> et </markdown>.
  4. Ajouter le résultat au contexte pour la page suivante.

Prérequis :
  - Variable d'environnement UNISTRA_API_KEY définie.
  - PyMuPDF (fitz) pour la conversion PDF → images.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import requests

# ── Configuration UNISTRA Qwen ────────────────────────────────────────
UNISTRA_URL = "https://conversation.ia.unistra.fr/api/v1/chat/completions"
UNISTRA_MODEL = "chat-qwen"


def parse_existing_md(md_path: Path) -> tuple[set[int], int, int]:
    """
    Parse un .md existant pour extraire :
    - Les pages déjà traitées (set d'indices).
    - Le nombre total de pages (d'après les métadonnées).
    - Le dernier image_counter utilisé (via les références image_XXX.jpg).
    """
    if not md_path.exists():
        return set(), 0, 1

    content = md_path.read_text(encoding="utf-8")

    # Extraire les pages traitées et leur statut
    pages_traitées = set()
    total_pages = 0
    for match in re.finditer(r"<!-- Page (\d+)/(\d+) : OK -->", content):
        page_num = int(match.group(1))
        total_pages = max(total_pages, int(match.group(2)))
        pages_traitées.add(page_num)

    # Extraire le dernier image_counter (ex: image_003.jpg → 3)
    last_image_num = 0
    for match in re.finditer(r"image_(\d{3})\.jpg", content):
        last_image_num = max(last_image_num, int(match.group(1)))
    last_image_counter = last_image_num + 1  # Prochain ID disponible

    return pages_traitées, total_pages, last_image_counter


def pdf_pages_to_base64(pdf_path: Path, dpi: int = 150) -> list[str]:
    """Convertir chaque page d'un PDF en image JPEG encodée en base64."""
    doc = fitz.open(str(pdf_path))
    images: list[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("jpeg")
        images.append(base64.b64encode(img_bytes).decode("utf-8"))
    doc.close()
    return images


def ask_llm(
    image_b64: str,
    accumulated_context: str,
    system_prompt: str,
    api_key: str,
    timeout: int = 600,
) -> str | None:
    """Envoyer une page au LLM Qwen et retourner le contenu de la réponse.

    Retourne None si la réponse ne contient pas de balises <markdown>.
    """
    user_content: list[dict] = [{"type": "text", "text": ""}]

    # Contexte des pages précédentes
    if accumulated_context:
        user_content.append({
            "type": "text",
            "text": (
                "CONTEXTE DES PAGES PRÉCÉDENTES (à ne PAS répéter) :\n"
                "<CONTEXT>\n"
                f"{accumulated_context}\n"
                "</CONTEXT>"
            ),
        })

    # Image de la page courante
    user_content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
    })

    payload = {
        "model": UNISTRA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(UNISTRA_URL, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    return content


def extract_markdown(response: str | None) -> str | None:
    """Extraire le texte entre <markdown> et </markdown>.

    Retourne None si les balises sont absentes.
    """
    if response is None:
        return None
    start = response.find("<markdown>")
    end = response.find("</markdown>")
    if start >= 0 and end > start:
        return response[start + len("<markdown>"):end].strip()
    return None


def convert_pdf(
    pdf_path: Path,
    api_key: str,
    timeout: int = 600,
    missing_pages: list[int] | None = None,
    initial_image_counter: int = 1,
) -> bool:
    """Convertir un PDF en Markdown via l'API UNISTRA Qwen (avec reprise et métadonnées).

    Args:
        pdf_path: Chemin vers le PDF.
        api_key: Clé API UNISTRA.
        timeout: Timeout par page (secondes).
        missing_pages: Liste des pages à traiter (None = tout convertir).
        initial_image_counter: Compteur initial pour les images.

    Retourne True en cas de succès.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    stem = pdf_path.stem
    md_path = pdf_path.parent / f"{stem}.md"
    images_dir = pdf_path.parent / f"{stem}_images"

    # 1. Conversion PDF → images
    print(f"[info] Conversion de {pdf_path.name} en images...")
    images_b64 = pdf_pages_to_base64(pdf_path)
    num_pages = len(images_b64)
    print(f"[info] {num_pages} page(s) détectée(s)")

    # 2. System prompt
    images_subdir = f"{stem}_images"
    system_prompt = (
        "Tu es un expert en transcription fidèle de documents PDF.\n"
        "Pour chaque page :\n"
        "1. Transcris EXACTEMENT tout le texte visible, sans rien changer ni corriger.\n"
        "2. Pour CHAQUE image, schéma, graphique, figure, photo ou illustration visible,\n"
        "   insère à l'endroit exact où elle apparaît dans le flux le bloc suivant :\n"
        "   | ![Courte description de l'image](IMAGES_DIR/IMAGE_ID) |\n"
        "   |:--:|\n"
        "   | *Description exhaustive du contenu visuel : sujet, couleurs, données\n"
        "   représentées (si graphique), tendances, échelles, tout élément pertinent.* |\n"
        "   - Le alt text entre crochets doit être COURT (5-10 mots max).\n"
        "   - IMAGE_ID : l'identifiant fourni dans le contexte (ex: image_003.jpg).\n"
        f"   - IMAGES_DIR : {images_subdir}\n"
        "   - La description en italique doit être détaillée.\n"
        "3. Retourne UNIQUEMENT le résultat entre <markdown> et </markdown>.\n"
        "Si la page est vide, retourne <markdown></markdown>."
    )

    # 3. Déterminer la liste des pages à traiter
    if missing_pages is None:
        # Nouvelle conversion : toutes les pages
        pages_to_process = list(range(1, num_pages + 1))
    else:
        pages_to_process = missing_pages

    # 4. Charger l'état existant depuis le .md (toutes les pages OK, dans l'ordre)
    all_transcriptions: list[str] = []
    pages_content: dict[int, str] = {}  # page_num → contenu (sans métadonnées)

    if md_path.exists() and missing_pages is not None:
        existing_content = md_path.read_text(encoding="utf-8")
        for match in re.finditer(r"<!-- Page (\d+)/\d+ : OK -->\s*(.*?)(?=<!-- Page |$)", existing_content, re.DOTALL):
            page_num = int(match.group(1))
            content = match.group(2).strip()
            pages_content[page_num] = content
            if content:
                all_transcriptions.append(content)
    elif md_path.exists() and missing_pages is None:
        # Force : écraser
        md_path.write_text("", encoding="utf-8")  # Sera surchargé au fur et à mesure

    image_counter = initial_image_counter

    # 5. Traitement des pages
    for page_idx in pages_to_process:
        image_b64 = images_b64[page_idx - 1]  # Index 0-based
        print(f"  Page {page_idx}/{num_pages}...")

        # --- Construction du contexte ---
        context_parts = []
        if len(all_transcriptions) > 5:
            num_previous = len(all_transcriptions) - 5
            context_parts.append(f"[Résumé : {num_previous} page(s) déjà traitées avant les 5 dernières]")

        start_idx = max(0, len(all_transcriptions) - 5)
        recent_pages = all_transcriptions[start_idx:]
        if recent_pages:
            context_parts.append("\n\n".join(recent_pages))

        context_for_llm = "\n\n".join(context_parts)
        if context_for_llm:
            context_for_llm += f"\n\nProchain identifiant d'image à utiliser : image_{image_counter:03d}.jpg"
        else:
            context_for_llm = f"Prochain identifiant d'image à utiliser : image_{image_counter:03d}.jpg"

        # --- Envoi à l'API avec retry ---
        transcription = None
        for attempt in range(2):
            response = ask_llm(
                image_b64=image_b64,
                accumulated_context=context_for_llm,
                system_prompt=system_prompt,
                api_key=api_key,
                timeout=timeout,
            )
            transcription = extract_markdown(response)
            if transcription is not None:
                break
            if attempt == 0:
                print(f"    ⚠ Tentative 1/2 : pas de balises <markdown>, réessai...")

        # --- Métadonnées et mise à jour ---
        if transcription is not None:
            if transcription:
                print(f"    ✓ Page {page_idx} extraite ({len(transcription)} car.)")
                img_count = transcription.count("| ![")
                if img_count:
                    print(f"    🖼  {img_count} image(s) décrite(s) (→ image_{image_counter + img_count:03d}.jpg)")
                    image_counter += img_count
            else:
                print(f"    ○ Page {page_idx} vide")
            pages_content[page_idx] = transcription
        else:
            print(f"    ✗ Page {page_idx} : échec après 2 tentatives")
            pages_content[page_idx] = None  # Marqueur ÉCHEC

        # --- Reconstruire all_transcriptions dans l'ordre ---
        all_transcriptions = []
        for pn in sorted(pages_content.keys()):
            c = pages_content[pn]
            if c is not None and isinstance(c, str):
                all_transcriptions.append(c)

        # --- Réécriture du .md complet (dans l'ordre des pages) ---
        md_content = ""
        for pn in sorted(pages_content.keys()):
            c = pages_content[pn]
            if md_content:
                md_content += "\n\n"
            if c is None:
                md_content += f"<!-- Page {pn}/{num_pages} : ÉCHEC (pas de balises markdown après 2 tentatives) -->"
            elif c == "":
                md_content += f"<!-- Page {pn}/{num_pages} : OK -->\n"
            else:
                md_content += f"<!-- Page {pn}/{num_pages} : OK -->\n{c}"
        md_path.write_text(md_content, encoding="utf-8")

    # 6. Dossier images
    images_dir.mkdir(exist_ok=True)
    print(f"[done] Dossier images créé : {images_dir}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convertir un PDF ou tous les PDF d'un dossier en Markdown via l'API UNISTRA Qwen."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Chemin du PDF d'entrée, ou d'un dossier à parcourir récursivement.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout en secondes par page (par défaut : 600).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forcer la reconversion complète (ignore les .md existants).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("UNISTRA_API_KEY")
    if not api_key:
        print("❌ UNISTRA_API_KEY manquante. Définis-la dans les variables d'environnement.", file=sys.stderr)
        return 1

    input_path = args.input
    if not input_path.exists():
        print(f"Erreur : {input_path} n'existe pas.", file=sys.stderr)
        return 1

    try:
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            # Mode fichier unique
            md_path = input_path.parent / f"{input_path.stem}.md"
            if md_path.exists() and not args.force:
                # Analyser le .md existant pour la reprise
                pages_traitées, _, last_image_counter = parse_existing_md(md_path)
                all_pages = set(range(1, len(pdf_pages_to_base64(input_path)) + 1))
                missing_pages = sorted(all_pages - pages_traitées)

                if not missing_pages:
                    print(f"Skip (déjà converti) : {md_path}")
                    return 0
                else:
                    print(f"[info] Reprise de {input_path.name} : pages manquantes {missing_pages}")
                    convert_pdf(
                        pdf_path=input_path,
                        api_key=api_key,
                        timeout=args.timeout,
                        missing_pages=missing_pages,
                        initial_image_counter=last_image_counter,
                    )
            else:
                convert_pdf(
                    pdf_path=input_path,
                    api_key=api_key,
                    timeout=args.timeout,
                )
        elif input_path.is_dir():
            # Mode batch : parcours récursif
            pdf_count = 0
            skip_count = 0
            for dirpath, _, filenames in os.walk(str(input_path)):
                for filename in filenames:
                    if filename.lower().endswith(".pdf"):
                        pdf_path = Path(dirpath) / filename
                        md_path = pdf_path.parent / f"{pdf_path.stem}.md"
                        pdf_count += 1

                        if md_path.exists() and not args.force:
                            pages_traitées, _, last_image_counter = parse_existing_md(md_path)
                            all_pages = set(range(1, len(pdf_pages_to_base64(pdf_path)) + 1))
                            missing_pages = sorted(all_pages - pages_traitées)

                            if not missing_pages:
                                print(f"Skip (déjà converti) : {md_path}")
                                skip_count += 1
                                continue
                            else:
                                print(f"[info] Reprise de {pdf_path.name} : pages manquantes {missing_pages}")
                                convert_pdf(
                                    pdf_path=pdf_path,
                                    api_key=api_key,
                                    timeout=args.timeout,
                                    missing_pages=missing_pages,
                                    initial_image_counter=last_image_counter,
                                )
                        else:
                            convert_pdf(
                                pdf_path=pdf_path,
                                api_key=api_key,
                                timeout=args.timeout,
                            )
            print(f"\nTerminé. {pdf_count} PDF(s) trouvé(s), {skip_count} déjà converti(s).")
        else:
            print(f"Erreur : {input_path} n'est ni un PDF ni un dossier.", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
