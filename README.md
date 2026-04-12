# llmtools

Boîte à outils pour expérimentations LLM, organisée en deux volets :

- `mcptools` : serveurs MCP locaux (Wikipedia, StackOverflow, recherche/SearXNG) exposés via `mcp-proxy`.
- `pdftools` : conversion de PDF vers Markdown avec extraction d'images (en local avec Docling/Granite ou via l'API Mistral).

## Structure

```text
llmtools/
├── mcptools/
│   ├── config-mcp.json
│   ├── README.md
│   └── start-mcp.sh
└── pdftools/
    ├── docling-pdf2md.py
    ├── mistral-pdf2md.py
    ├── README.md
    ├── requirements.txt
    └── test/
```

## 1) Outils MCP (`mcptools`)

Configuration locale de 3 serveurs MCP, tous encapsulés par `mcp-trunc-proxy` pour limiter la taille des réponses :

- `wikipedia` (langue `fr`)
- `stackoverflow`
- `search` (via SearXNG local)

### Prérequis

- `uvx`
- `npx` (Node.js)

### Démarrage

```bash
cd mcptools
chmod +x start-mcp.sh
./start-mcp.sh
```

Le script :

1. Démarre SearXNG (`simplexng`) en arrière-plan.
2. Lance `mcp-proxy` avec `config-mcp.json` sur le port `8001`.
3. Arrête proprement SearXNG quand `mcp-proxy` est stoppé.

### Points d'acces MCP

- `http://127.0.0.1:8001/servers/wikipedia/mcp`
- `http://127.0.0.1:8001/servers/stackoverflow/mcp`
- `http://127.0.0.1:8001/servers/search/mcp`

Test direct de SearXNG : `http://127.0.0.1:8888`

Voir aussi : `mcptools/README.md`

## 2) Outils PDF (`pdftools`)

Deux scripts de conversion PDF vers Markdown :

- `docling-pdf2md.py` : conversion locale (Docling + Granite VLM), adaptée aux documents privés.
- `mistral-pdf2md.py` : conversion via l'API Mistral OCR, rapide mais nécessitant une clé d'API.

### Installation

```bash
cd pdftools
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Utilisation de Docling (local)

```bash
python docling-pdf2md.py <chemin_pdf> -o <dossier_sortie>
```

Exemple :

```bash
python docling-pdf2md.py test/sample.pdf -o test/docling_output
```

### Utilisation de Mistral OCR (API)

Configurer la clé :

```bash
export MISTRAL_API_KEY="votre-cle-api"
```

Lancer la conversion récursive d'un dossier :

```bash
python mistral-pdf2md.py <dossier>
```

Exemple :

```bash
python mistral-pdf2md.py test
```

Voir aussi : `pdftools/README.md`

## Choisir le bon outil PDF

- Priorité confidentialité / hors ligne : Docling.
- Priorité simplicité / OCR distant : Mistral.

## Documentation détaillée

- MCP : `mcptools/README.md`
- PDF : `pdftools/README.md`