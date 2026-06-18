#!/usr/bin/env python3
"""Test end-to-end de unistra-pdf2md.py.

1. Crée un fichier Markdown bidon.
2. Le convertit en PDF (via PyMuPDF).
3. Lance unistra-pdf2md.py sur ce PDF.
4. Vérifie que le fichier .md de sortie existe et n'est pas vide.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "unistra-pdf2md.py"
TEST_DIR = Path(__file__).resolve().parent / "test_output"

# ── Markdown bidon ───────────────────────────────────────────────────
SAMPLE_MD = """# Test Document

## Section 1

Voici un paragraphe de test pour vérifier la conversion PDF → Markdown.
Le texte doit être **fidèle** et ne pas être modifié.

### Liste à puces

- Premier élément
- Deuxième élément
- Troisième élément

### Tableau

| Colonne A | Colonne B |
|-----------|-----------|
| Valeur 1  | Valeur X  |
| Valeur 2  | Valeur Y  |

## Section 2

Un deuxième paragraphe avec du texte *italique* et du **gras**.

---

Fin du document de test.
"""


def create_sample_md() -> Path:
    """Écrire le fichier Markdown de test."""
    test_dir = TEST_DIR
    test_dir.mkdir(parents=True, exist_ok=True)

    md_path = test_dir / "test_sample.md"
    md_path.write_text(SAMPLE_MD, encoding="utf-8")
    print(f"[ok] Sample MD créé : {md_path}")
    return md_path


def md_to_pdf(md_path: Path) -> Path:
    """Convertir un Markdown en PDF (via PyMuPDF)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    # insérer du texte brut sur la page
    page.insert_text(
        (72, 72),
        "Test Document\n\nSection 1\n\nVoici un paragraphe de test pour vérifier la conversion PDF → Markdown.\nLe texte doit être fidèle et ne pas être modifié.\n\nListe à puces\n\n- Premier élément\n- Deuxième élément\n- Troisième élément\n\nSection 2\n\nUn deuxième paragraphe avec du texte italique et du gras.\n\nFin du document de test.",
        fontsize=12,
    )
    pdf_path = md_path.parent / "test_sample.pdf"
    doc.save(str(pdf_path))
    doc.close()
    print(f"[ok] PDF créé : {pdf_path}")
    return pdf_path


def run_conversion(pdf_path: Path) -> Path:
    """Lancer unistra-pdf2md.py et retourner le chemin du .md."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(pdf_path),
    ]
    print(f"[info] Lancement : {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        print(f"[err] Le script a échoué (code {result.returncode})", file=sys.stderr)
        sys.exit(1)

    # Le .md est écrit à côté du PDF
    md_output = pdf_path.parent / f"{pdf_path.stem}.md"
    if not md_output.exists():
        print(f"[err] {md_output} n'a pas été créé", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] Sortie Markdown : {md_output}")
    return md_output


def validate(md_path: Path) -> None:
    """Vérifier que le Markdown de sortie n'est pas vide."""
    content = md_path.read_text(encoding="utf-8")
    if not content.strip():
        print("[err] Le Markdown de sortie est vide.", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] Markdown de sortie : {len(content)} caractères")
    print("=== Contenu ===")
    print(content[:500])
    if len(content) > 500:
        print("...")
    print("===============")


def create_single_page_pdf(output_dir: Path, name: str) -> Path:
    """Créer un PDF d'une seule page avec du texte simple."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        f"Document {name}\n\nCeci est le contenu du document {name}.\nIl contient un paragraphe de test pour vérifier\nla conversion PDF→Markdown.\n\nFin du document {name}.",
        fontsize=12,
    )
    pdf_path = output_dir / f"{name}.pdf"
    doc.save(str(pdf_path))
    doc.close()
    print(f"[ok] PDF créé : {pdf_path}")
    return pdf_path


def create_3page_pdf(output_dir: Path) -> Path:
    """Créer un PDF de 3 pages avec du texte distinct sur chaque page."""
    import fitz

    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "PAGE 1 — Introduction\n\nCe document de test contient trois pages.\nChaque page a un contenu distinct pour vérifier\nque la conversion PDF→Markdown fonctionne\ncorrectement sur un document multi-pages.\n\nLa première page introduit le sujet.", fontsize=12)

    page2 = doc.new_page()
    page2.insert_text((72, 72), "PAGE 2 — Développement\n\nVoici le corps du document de test.\nIl contient plusieurs paragraphes et des\nlistes à puces pour vérifier la structure.\n\n- Élément A\n- Élément B\n- Élément C\n\nUn tableau également :\n\n| Colonne 1 | Colonne 2 |\n|-----------|-----------|\n| Valeur 1  | Valeur X  |\n| Valeur 2  | Valeur Y  |", fontsize=12)

    page3 = doc.new_page()
    page3.insert_text((72, 72), "PAGE 3 — Conclusion\n\nCe document de test arrive à sa fin.\nLes trois pages ont été converties en\nimages puis transcrites en Markdown\npar l'API UNISTRA Qwen.\n\nFin du test.", fontsize=12)

    pdf_path = output_dir / "test_3pages.pdf"
    doc.save(str(pdf_path))
    doc.close()
    print(f"[ok] PDF 3 pages créé : {pdf_path}")
    return pdf_path


def run_conversion_multi(pdf_path: Path, name: str, expected_pages: int) -> Path:
    """Lancer unistra-pdf2md.py sur un PDF multi-pages et valider."""
    cmd = [
        sys.executable,
        str(SCRIPT),
        str(pdf_path),
    ]
    print(f"[info] Lancement ({name}): {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    if result.returncode != 0:
        print(f"[err] Le script a échoué (code {result.returncode})", file=sys.stderr)
        sys.exit(1)

    # Le .md est écrit à côté du PDF
    md_output = pdf_path.parent / f"{pdf_path.stem}.md"
    if not md_output.exists():
        print(f"[err] {md_output} n'a pas été créé", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] Sortie Markdown ({name}) : {md_output}")
    return md_output


def validate_multi_pages(md_path: Path, expected_pages: int) -> None:
    """Vérifier que le Markdown contient bien le contenu de toutes les pages."""
    content = md_path.read_text(encoding="utf-8")
    if not content.strip():
        print("[err] Le Markdown de sortie est vide.", file=sys.stderr)
        sys.exit(1)

    print(f"[ok] Markdown de sortie : {len(content)} caractères")

    # Vérifier la présence de repères de pages
    page_markers = []
    for i in range(1, expected_pages + 1):
        if f"PAGE {i}" in content:
            page_markers.append(i)

    if len(page_markers) == expected_pages:
        print(f"[ok] Toutes les {expected_pages} pages détectées dans le Markdown")
    else:
        print(f"[err] Seules {len(page_markers)}/{expected_pages} pages détectées")
        sys.exit(1)

    print("=== Contenu (extrait) ===")
    print(content[:600])
    if len(content) > 600:
        print("...")
    print("========================")


def main() -> None:
    # Vérifier UNISTRA_API_KEY
    if not os.environ.get("UNISTRA_API_KEY"):
        print("⚠️  UNISTRA_API_KEY n'est pas définie — test en mode SKIP (sans appel API).", file=sys.stderr)
        print("Pour un test complet, définis UNISTRA_API_KEY puis relance : python test_unistra.py", file=sys.stderr)
        sys.exit(0)

    # ── Test 1 : PDF 1 page ──────────────────────────────────────────
    print("\n═══ Test 1 : PDF 1 page ═══")
    md_path = create_sample_md()
    pdf_path = md_to_pdf(md_path)
    md_output = run_conversion(pdf_path)
    validate(md_output)
    print("✅ Test 1 réussi !")

    # ── Test 2 : PDF 3 pages ─────────────────────────────────────────
    print("\n═══ Test 2 : PDF 3 pages ═══")
    pdf_3pages = create_3page_pdf(TEST_DIR)
    md_3pages = run_conversion_multi(pdf_3pages, "3pages", expected_pages=3)
    validate_multi_pages(md_3pages, expected_pages=3)
    print("✅ Test 2 réussi !")

    # ── Test 3 : Batch mode (dossier avec plusieurs PDFs) ────────────
    print("\n═══ Test 3 : Batch mode dossier ═══")
    batch_dir = TEST_DIR / "batch_test"
    batch_dir.mkdir(exist_ok=True)

    # Créer 3 PDFs dans le dossier batch
    pdfs = []
    for i in range(1, 4):
        pdf_path = create_single_page_pdf(batch_dir, f"doc_{i}")
        pdfs.append(pdf_path)
        print(f"  Créé : {pdf_path.name}")

    # 1ère exécution : tous les PDFs sont convertis
    print("\n  --- 1ère exécution ---")
    cmd = [sys.executable, str(SCRIPT), str(batch_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    if result.returncode != 0:
        print(f"[err] Échec 1ère exécution (code {result.returncode})", file=sys.stderr)
        sys.exit(1)

    # Vérifier que tous les .md existent
    for pdf in pdfs:
        md_expected = pdf.parent / f"{pdf.stem}.md"
        if not md_expected.exists():
            print(f"[err] {md_expected} attendu mais inexistant", file=sys.stderr)
            sys.exit(1)
    print(f"[ok] Tous les {len(pdfs)} PDFs ont été convertis")

    # 2ème exécution : tout doit être skipé
    print("\n  --- 2ème exécution (skip) ---")
    cmd = [sys.executable, str(SCRIPT), str(batch_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")

    skip_count = result.stdout.count("Skip")
    if skip_count >= len(pdfs):
        print(f"[ok] {skip_count} PDF(s) skipé(s) — tout était déjà converti")
    else:
        print(f"[err] Seuls {skip_count}/{len(pdfs)} PDFs skipés")
        sys.exit(1)

    # Supprimer les .md pour re-tester que le batch fonctionne proprement
    for pdf in pdfs:
        md_path = pdf.parent / f"{pdf.stem}.md"
        if md_path.exists():
            md_path.unlink()
        images_dir = pdf.parent / f"{pdf.stem}_images"
        if images_dir.exists():
            images_dir.rmdir()

    print("\n✅ Test 3 réussi !")

    print("\n✅ Tous les tests sont passés !")


if __name__ == "__main__":
    main()
