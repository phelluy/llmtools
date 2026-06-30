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
import sys
from pathlib import Path

import fitz  # PyMuPDF
import requests

# ── Configuration UNISTRA Qwen ────────────────────────────────────────
UNISTRA_URL = "https://conversation.ia.unistra.fr/api/v1/chat/completions"
UNISTRA_MODEL = "chat-qwen"


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


def convert_pdf(pdf_path: Path, api_key: str, timeout: int = 600) -> bool:
    """Convertir un PDF en Markdown via l'API UNISTRA Qwen (incrémental).

    Le .md est écrit au même niveau que le PDF.
    Un sous-dossier images/ est créé (vide pour l'instant).

    Retourne True en cas de succès, False sinon.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    stem = pdf_path.stem
    md_path = pdf_path.parent / f"{stem}.md"
    images_dir = pdf_path.parent / f"{stem}_images"

    # 1. Convertir les pages en images
    print(f"[info] Conversion de {pdf_path.name} en images...")
    images_b64 = pdf_pages_to_base64(pdf_path)
    num_pages = len(images_b64)
    print(f"[info] {num_pages} page(s) détectée(s)")

    # 2. Système prompt
    # Note : <markdown>...</markdown> n'est pas du GFM — c'est une balise interne
    # utilisée pour extraire la réponse du LLM, strippée avant l'écriture du .md.
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
        "   - Le alt text entre crochets doit être COURT (5-10 mots max) et décrire\n"
        "     brièvement l'image (ex: « Logo Python bleu et jaune »)\n"
        "   - IMAGE_ID : l'identifiant fourni dans le contexte (ex: image_003.jpg)\n"
        f"   - IMAGES_DIR : {images_subdir}\n"
        "   - La description en italique doit être assez détaillée pour comprendre\n"
        "     l'image sans la voir, même si le fichier est absent.\n"
        "3. Retourne UNIQUEMENT le résultat entre <markdown> et </markdown>.\n"
        "Si la page est vide, retourne <markdown></markdown>."
    )

    # 3. Traitement incrémental page par page
    accumulated_text = ""
    image_counter = 1
    page_results: list[tuple[int, str]] = []  # (page_num, transcription)

    for page_idx, image_b64 in enumerate(images_b64, start=1):
        print(f"  Page {page_idx}/{num_pages}...")

        # Construire le contexte avec le compteur d'images
        context_for_llm = accumulated_text
        context_for_llm += (
            f"\n\nProchain identifiant d'image à utiliser : image_{image_counter:03d}.jpg"
        )

        response = ask_llm(
            image_b64=image_b64,
            accumulated_context=context_for_llm,
            system_prompt=system_prompt,
            api_key=api_key,
            timeout=timeout,
        )
        transcription = extract_markdown(response)

        if transcription is not None:
            if transcription:
                print(f"    ✓ Page {page_idx} extraite ({len(transcription)} car.)")
                # Mettre à jour le compteur en comptant les images référencées
                img_count = transcription.count("| ![")
                if img_count:
                    print(f"    🖼  {img_count} image(s) décrite(s) (→ image_{image_counter + img_count:03d}.jpg)")
                    image_counter += img_count
            else:
                print(f"    ○ Page {page_idx} vide")
            page_results.append((page_idx, transcription))
            accumulated_text += "\n" + transcription if accumulated_text else transcription
        else:
            print(f"    ✗ Page {page_idx} : pas de balises <markdown> dans la réponse")
            page_results.append((page_idx, ""))

    # 4. Écrire le markdown final
    final_md = "\n\n".join(text for _, text in page_results)
    md_path.write_text(final_md, encoding="utf-8")
    print(f"[done] Markdown écrit : {md_path}")

    # 5. Créer le dossier images (vide pour l'instant)
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
                        if md_path.exists():
                            print(f"Skip (déjà converti) : {md_path}")
                            skip_count += 1
                            continue
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
