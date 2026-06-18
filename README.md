# llmtools

Boîte à outils pour expérimentations LLM, organisée en deux volets :

- `mcptools` : serveurs MCP locaux (Python, Wikipedia, StackOverflow, recherche/SearXNG) exposés via `mcp-proxy`.
- `pdftools` : conversion de PDF vers Markdown (Mistral OCR API et UNISTRA Qwen).

## Structure

```text
llmtools/
├── mcptools/
│   ├── config-mcp.json
│   ├── README.md
│   └── start-mcp.sh
└── pdftools/
    ├── mistral-pdf2md.py
    ├── unistra-pdf2md.py
    ├── requirements.txt
    └── test/
```

## 1) Outils MCP (`mcptools`)

Configuration locale de 4 serveurs MCP :

- `wikipedia` (langue `fr`)
- `stackoverflow`
- `search` (via SearXNG local)
- `python` (interpréteur Python MCP, avec bibliothèques scientifiques)

Les serveurs `wikipedia`, `stackoverflow` et `search` sont encapsulés par `mcp-trunc-proxy` pour limiter la taille des réponses.

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
- `http://127.0.0.1:8001/servers/python/mcp`

Test direct de SearXNG : `http://127.0.0.1:8888`

Note : le serveur `python` utilise `mcp-python-interpreter` avec un environnement virtuel pointant vers un chemin local externe au dépôt (défini dans `mcptools/config-mcp.json`).

Voir aussi : `mcptools/README.md`

## 2) Outils PDF (`pdftools`)

Deux scripts de conversion PDF vers Markdown :

- `mistral-pdf2md.py` : conversion via l'API Mistral OCR.
- `unistra-pdf2md.py` : conversion via l'API UNISTRA (Qwen vision), avec accumulation de contexte page par page.

### Installation

```bash
cd pdftools
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
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

### UNISTRA Qwen

Configurer la clé :

```bash
export UNISTRA_API_KEY="votre-cle-api"
```

Lancer la conversion :

```bash
python unistra-pdf2md.py <fichier_ou_dossier>
```

Exemple :

```bash
python unistra-pdf2md.py test
```

## Documentation

- MCP : `mcptools/README.md`
- PDF : `pdftools/README.md`