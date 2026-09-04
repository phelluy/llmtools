# llmtools

Boîte à outils pour expérimentations LLM, organisée en deux volets :

- `mcptools` : serveurs MCP locaux (Python, Wikipedia, recherche/SearXNG) exposés via `mcp-proxy`.
- `pdftools` : conversion de PDF vers Markdown (Mistral OCR API et UNISTRA Qwen).

## Structure

```text
llmtools/
├── mcptools/
│   ├── config-mcp.json          (repli)
│   ├── config-mcp-macos.json
│   ├── config-mcp-linux.json
│   ├── mcp_python_server.py
│   ├── README.md
│   ├── start-mcp.sh
│   └── workdir/                 (.venv/, scripts/, requirements.txt)
└── pdftools/
    ├── mistral-pdf2md.py
    ├── unistra-pdf2md.py
    ├── requirements.txt
    └── test/
```

## 1) Outils MCP (`mcptools`)

Configuration locale de 3 serveurs MCP :

- `wikipedia` (langue `fr`)
- `search` (via SearXNG local)
- `python` (interpréteur Python MCP, avec bibliothèques scientifiques)

Les serveurs `wikipedia` et `search` sont encapsulés par `mcp-trunc-proxy` pour limiter la taille des réponses.

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
2. Lance `mcp-proxy` avec la config adaptée à l'OS (`config-mcp-macos.json` / `config-mcp-linux.json`, `config-mcp.json` en repli) sur le port `8001`.
3. Arrête proprement SearXNG quand `mcp-proxy` est stoppé.

### Points d'acces MCP

- `http://127.0.0.1:8001/servers/wikipedia/mcp`
- `http://127.0.0.1:8001/servers/search/mcp`
- `http://127.0.0.1:8001/servers/python/mcp`

Test direct de SearXNG : `http://127.0.0.1:8888`

Note : le serveur `python` utilise `mcp_python_server.py` (serveur FastMCP maison, natif SDK MCP 2.x) avec un venv local au dépôt (`mcptools/workdir/.venv`, créé automatiquement par `start-mcp.sh` si absent). `run_python_code` tourne dans l'env `uvx` (libs `--with`) ; `run_python_file` dans le venv local (libs via `workdir/requirements.txt`).

Voir aussi : `mcptools/README.md`

## 2) Outils PDF (`pdftools`)

Trois outils PDF :

- `mistral-pdf2md.py` : conversion via l'API Mistral OCR.
- `unistra-pdf2md.py` : conversion via l'API UNISTRA (Qwen vision), avec accumulation de contexte page par page et reprise après interruption.
- `check-extraction.py` : vérification de la qualité d'extraction en comparant le texte brut extrait (PyMuPDF) avec le markdown généré (coefficient de Jaccard par page).

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