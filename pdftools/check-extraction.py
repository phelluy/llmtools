#!/usr/bin/env python3
"""Vérifie la fidélité de l'extraction markdown (unistra-pdf2md.py) en comparant
le texte brut du PDF (PyMuPDF) avec le contenu extrait par le LLM.

Usage:
    python3 check-extraction.py monpdf.pdf [--threshold 0.3]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


def clean_text(text: str) -> str:
    """Supprime la ponctuation et normalise les espaces."""
    # Garder lettres, chiffres et apostrophes simples
    text = re.sub(r"[^\w'\u00e0-\u00ff]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    """Tokenise un texte nettoyé en ensemble de mots (unigrammes)."""
    return set(clean_text(text).split())


def jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Coefficient de Jaccard entre deux ensembles."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_raw_text(pdf_path: Path) -> list[str]:
    """Extrait le texte brut de chaque page du PDF (PyMuPDF)."""
    doc = fitz.open(str(pdf_path))
    pages_text: list[str] = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()
    return pages_text


def extract_markdown_text(md_path: Path) -> dict[int, str]:
    """Parse le .md généré par unistra-pdf2md.py.

    Retourne {page_num: texte_markdown} pour les pages OK.
    Supprime les descriptions d'images entre |:--:| et le | suivant.
    """
    content = md_path.read_text(encoding="utf-8")
    pages: dict[int, str] = {}

    for match in re.finditer(
        r"<!-- Page (\d+)/\d+ : OK -->\s*(.*?)(?=<!-- Page |$)",
        content,
        re.DOTALL,
    ):
        page_num = int(match.group(1))
        text = match.group(2).strip()

        # Supprimer les blocs d'images markdown (| ![...](...) | ... |)
        text = re.sub(
            r"\| !\[.*?\]\(.*?\) \|.*?\|:--:\|.*?\n\| \*.*?\* \|",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(r"\| !\[.*?\]\(.*?\) \|", "", text)

        # Supprimer les commentaires HTML
        text = re.sub(r"<!--.*?-->", "", text)

        pages[page_num] = text.strip()

    return pages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare le texte brut du PDF avec l'extraction markdown du LLM."
    )
    parser.add_argument("pdf", type=Path, help="Chemin vers le PDF.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Seuil Jaccard en dessous duquel une page est signalée (défaut: 0.3).",
    )
    parser.add_argument(
        "--md",
        type=Path,
        default=None,
        help="Chemin vers le .md (défaut: même nom que le PDF).",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    md_path = args.md or pdf_path.parent / f"{pdf_path.stem}.md"

    if not pdf_path.exists():
        print(f"Erreur : PDF introuvable : {pdf_path}", file=sys.stderr)
        return 1
    if not md_path.exists():
        print(f"Erreur : .md introuvable : {md_path}", file=sys.stderr)
        return 1

    # Extraction
    raw_pages = extract_raw_text(pdf_path)
    md_pages = extract_markdown_text(md_path)

    num_pages = len(raw_pages)
    print(f"[info] {num_pages} page(s) dans le PDF")
    print(f"[info] {len(md_pages)} page(s) OK dans le .md")
    print()

    # Comparaison page par page
    results: list[dict] = []
    alerts = 0

    for i in range(1, num_pages + 1):
        raw = raw_pages[i - 1]
        raw_tokens = tokenize(raw)
        raw_count = len(raw_tokens)

        md_text = md_pages.get(i, "")
        md_tokens = tokenize(md_text)
        md_count = len(md_tokens)

        score = jaccard(raw_tokens, md_tokens)
        status = "OK"

        if raw_count == 0 and md_count == 0:
            status = "OK (page vide des deux côtés)"
        elif raw_count == 0:
            status = "⚠️  TEXTE ABSENT (PDF)"
            alerts += 1
        elif md_count == 0:
            status = "⚠️  TEXTE ABSENT (extraction)"
            alerts += 1
        elif score < args.threshold:
            status = f"⚠️  FAIBLE"
            alerts += 1

        results.append({
            "page": i,
            "score": score,
            "raw_count": raw_count,
            "md_count": md_count,
            "status": status,
        })

        # Affichage par page
        print(
            f"  Page {i:3d}/{num_pages} : {status:35s} "
            f"Jaccard={score:.3f}  brut={raw_count}  extrait={md_count}"
        )

    # Compte rendu final
    print()
    total = len(results)
    ok_count = total - alerts
    print(f"─── Compte rendu ───")
    print(f"  Pages analysées : {total}")
    print(f"  Pages OK        : {ok_count}")
    print(f"  Alertes         : {alerts}")
    if alerts > 0:
        print(f"  Seuil           : {args.threshold:.2f}")
        print()
        print(f"  Pages à vérifier :")
        for r in results:
            if "⚠️" in r["status"]:
                print(
                    f"    Page {r['page']:3d} : {r['status']}  "
                    f"(Jaccard={r['score']:.3f})"
                )
        print()
        if alerts == total:
            print(
                "  ⚠️  Aucune page ne correspond au texte brut. "
                "Le PDF est peut-être un scan sans texte sélectionnable."
            )

    return 1 if alerts > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
