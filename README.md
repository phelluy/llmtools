# llmtools

Boîte à outils pour expérimentations LLM, organisée en deux volets :

- `mcptools` : serveurs MCP locaux (Wikipedia, recherche/SearXNG, Python, Playwright, shell) exposés via `mcp-proxy`.
- `pdftools` : conversion de PDF vers Markdown (Mistral OCR API et UNISTRA Qwen), avec vérification de la qualité d'extraction.

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
│   ├── test-formulaire/         (test end-to-end du serveur playwright)
│   ├── test_shell_driver.py     (test stdio de run_shell_command)
│   └── workdir/                 (espace de travail du serveur python, non versionné)
└── pdftools/
    ├── mistral-pdf2md.py
    ├── unistra-pdf2md.py
    ├── check-extraction.py      (vérification qualité d'extraction)
    ├── README.md
    ├── requirements.txt
    ├── test/                    (PDF et échantillons d'exemple)
    └── tests/                   (test end-to-end de unistra-pdf2md)
```

## 1) Outils MCP (`mcptools`)

Configuration locale de 5 serveurs MCP :

- `wikipedia` (langue `fr`)
- `search` (via SearXNG local)
- `python` (interpréteur Python MCP, avec bibliothèques scientifiques et outil shell `run_shell_command`)
- `playwright` (automatisation de navigateur Chrome : navigation, remplissage de formulaires, cases à cocher, clics)
- `shell` (exécution de commandes en argv avec allowlist, via `mcp-shell-server` de tumf)

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

1. Vérifie si SearXNG (`simplexng`) tourne déjà sur le port `8888` (le réutilise ou le démarre en arrière-plan).
2. Lance `mcp-proxy` avec la config adaptée à l'OS (`config-mcp-macos.json` / `config-mcp-linux.json`, `config-mcp.json` en repli) sur le port `8001` (avec origines configurables via `MCP_ALLOW_ORIGINS`).
3. Arrête proprement SearXNG quand `mcp-proxy` est stoppé (uniquement s'il a été lancé par le script).

### Points d'acces MCP

- `http://127.0.0.1:8001/servers/wikipedia/mcp`
- `http://127.0.0.1:8001/servers/search/mcp`
- `http://127.0.0.1:8001/servers/python/mcp`
- `http://127.0.0.1:8001/servers/playwright/mcp`
- `http://127.0.0.1:8001/servers/shell/mcp`

Test direct de SearXNG : `http://127.0.0.1:8888`

### Test du serveur playwright

Test end-to-end du remplissage de formulaire (texte + cases à cocher) via Playwright MCP :

```bash
cd mcptools/test-formulaire
./run.sh
```

La sortie doit se terminer par `TEST REUSSI`. Voir `mcptools/README.md` pour le détail.

### Test du serveur python (accès shell)

Test stdio de l'outil `run_shell_command` (lancement direct, sans `mcp-proxy`) :

```bash
cd mcptools
python3 test_shell_driver.py
```

Vérifie : cas nominal, expansion de `~`, code retour, dossier inexistant, timeout avec kill du groupe de processus, tronquage de sortie, pipelines.

Note : le serveur `python` utilise `mcp_python_server.py` (serveur FastMCP maison, natif SDK MCP 2.x). Il expose 5 outils (`run_python_code`, `run_python_file`, `list_sandbox_files`, `read_sandbox_file`, `run_shell_command`) partageant un seul interpréteur dans l'env `uvx` (libs `--with` : sympy, numpy, scipy, matplotlib, pandas, seaborn, scikit-learn, requests) avec gestion propre des timeouts et protection contre les sorties excessives. Pas de venv séparé. L'accès shell (outil `run_shell_command` + serveur `shell` de tumf) est détaillé dans `mcptools/README.md`.

### Connexion des clients MCP

- **llama-server** (webui) : ajouter les points d'accès `/servers/<nom>/mcp` listés ci-dessus dans la rubrique MCP de l'interface web, avec `--webui-mcp-proxy` au lancement. Commande complète dans `mcptools/README.md`.
- **mistral-vibe** : déclarer le serveur voulu dans `~/.vibe/config.toml` (transport `streamable-http`). Seul `playwright` est utile dans Vibe, les autres serveurs étant redondants avec ses outils natifs. Détail dans `mcptools/README.md`.

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

### Vérification et tests

- `check-extraction.py` : compare le texte brut (PyMuPDF) avec le markdown généré (coefficient de Jaccard par page). Voir `pdftools/README.md`.
- Test end-to-end de `unistra-pdf2md.py` (nécessite `UNISTRA_API_KEY`) :

```bash
cd pdftools
python tests/test_unistra.py
```

## Documentation

- MCP : `mcptools/README.md`
- PDF : `pdftools/README.md`