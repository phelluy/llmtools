# PDF to Markdown Converters

Deux outils pour convertir des PDF en Markdown, plus un outil de vérification de la qualité d'extraction.

| Outil | Mécanisme | Reprise | Vérification |
|-------|-----------|---------|-------------|
| `mistral-pdf2md.py` | API Mistral OCR | Non | Non |
| `unistra-pdf2md.py` | API UNISTRA Qwen (vision, page par page) | Oui | Non |
| `check-extraction.py` | Comparaison texte brut vs. extraction LLM | - | Oui |

## Installation

```bash
pip install pymupdf requests
```

**Dépendances :**
- `pymupdf` — Conversion PDF → images, extraction de texte brut
- `requests` — Appels API HTTP

## Outils disponibles

### 1. Mistral OCR to Markdown (`mistral-pdf2md.py`)

Convertit un PDF en Markdown via l'API Mistral OCR. Rapide, extrait les images.

```bash
export MISTRAL_API_KEY="your-api-key"
python mistral-pdf2md.py <directory>
```

**Sortie :** `<nom>.md` + dossier `sample_images/` avec les images extraites.

### 2. UNISTRA Qwen to Markdown (`unistra-pdf2md.py`)

Convertit un PDF en Markdown via l'API UNISTRA Qwen (modèle `chat-qwen`). Approche **incrémentale page par page** : chaque page est envoyée au LLM avec le contexte des pages précédentes.

**Fonctionnalités clés :**
- **Reprise automatique** : si le `.md` existe déjà, seules les pages manquantes ou en échec sont retraitées
- **Écriture incrémentale** : chaque page est sauvegardée dès qu'elle est traitée (pas de perte en cas d'interruption)
- **Retry** : 2 tentatives par page en cas d'absence de balises `<markdown>`
- **Contexte optimisé** : seules les 5 dernières pages + un résumé sont envoyés au LLM (évite les dépassements de contexte sur les longs documents)
- **Métadonnées** : commentaires HTML `<!-- Page X/N : OK/ÉCHEC -->` dans le `.md` pour le suivi
- **`--force`** : pour forcer une reconversion complète

```bash
export UNISTRA_API_KEY="your-api-key"

# Fichier unique
python unistra-pdf2md.py document.pdf

# Dossier (récursif)
python unistra-pdf2md.py ./documents/

# Timeout par page personnalisé (défaut : 600s)
python unistra-pdf2md.py document.pdf --timeout 120

# Forcer une reconversion complète
python unistra-pdf2md.py document.pdf --force
```

**Sortie :** `<nom>.md` + dossier `<nom>_images/` (réservé pour extraction future).

**Exemple de reprise :**
```bash
# Première exécution interrompue → pages 1-8 extraites, crash page 9
python unistra-pdf2md.py document.pdf

# Seconde exécution → reprend à la page 9 automatiquement
python unistra-pdf2md.py document.pdf
```

### 3. Vérification d'extraction (`check-extraction.py`)

Compare le texte brut extrait du PDF (PyMuPDF `page.get_text()`) avec le contenu markdown généré par `unistra-pdf2md.py`. Détecte les **hallucinations graves** (divergences massives entre le texte réel et l'extraction LLM).

**Métrique :** coefficient de Jaccard par page. Une page est signalée si son score est inférieur au seuil (défaut : 0.3).

```bash
# Vérification avec le .md correspondant au PDF
python check-extraction.py document.pdf

# Seuil personnalisé
python check-extraction.py document.pdf --threshold 0.5

# .md différent du PDF
python check-extraction.py document.pdf --md autre_fichier.md
```

**Sortie exemple :**
```
[info] 14 page(s) dans le PDF
[info] 14 page(s) OK dans le .md

  Page   1/14 : OK       Jaccard=0.864  brut=272  extrait=235
  ...
  Page   4/14 : FAIBLE   Jaccard=0.004  brut=346  extrait=8

─── Compte rendu ───
  Pages analysées : 14
  Pages OK        : 13
  Alertes         : 1
  Pages à vérifier :
    Page   4 : FAIBLE  (Jaccard=0.004)
```

**Limites :**
- Ne fonctionne que si le PDF a du **texte sélectionnable** (PDF natif, pas un scan)
- Ne détecte pas les erreurs fines (noms propres, nombres) — focus sur les écarts massifs

## Comparaison des outils

| Feature | Mistral | UNISTRA |
|---------|---------|---------|
| Extraction d'images | Oui (fichiers réels) | Non (descriptions textuelles) |
| OCR | Oui | Oui |
| PDF scannés | Excellent | Excellent |
| Reprise après interruption | Non | Oui |
| Retry automatique | Non | Oui (2 tentatives) |
| Métadonnées de suivi | Non | Oui (commentaires HTML) |
| Coût | API payante | API payante |
| Vitesse | Appels réseau | Appels réseau |
| Contexte page par page | Non | Oui |

## Documentation

- [Mistral API Documentation](https://docs.mistral.ai/)
- [UNISTRA Qwen API](https://github.com/unistra/qwen-inference-api)
