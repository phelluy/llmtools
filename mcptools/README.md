# mcp-configs

## Comment ça marche avec mcp-trunc-proxy

Les serveurs MCP `wikipedia` et `search` sont lancés derrière `mcp-trunc-proxy`.

Le principe est le suivant :

1. Le client appelle `mcp-proxy` sur le port `8001`.
2. `mcp-proxy` route la requête vers le serveur nommé configuré dans le fichier `config-mcp*.json` de l'OS.
3. Ce serveur nommé passe d'abord par `mcp-trunc-proxy`, qui limite la taille des réponses (`max-bytes`, `preview-max-chars`, etc.).
4. `mcp-trunc-proxy` appelle ensuite le serveur MCP réel (`wikipedia-mcp` ou `mcp-searxng`).

Ce montage évite les réponses trop longues, réduit le bruit dans les sorties et garde des résultats plus stables côté client.

Configuration locale de serveurs MCP (Wikipedia, Search via SearXNG, Python et Playwright) exposés par `mcp-proxy` sur le port `8001`.

## Fichiers du dépôt

- `start-mcp.sh` : démarre `simplexng` en arrière-plan, puis lance `mcp-proxy` avec la config adaptée à l'OS (`config-mcp-macos.json` sur macOS, `config-mcp-linux.json` sur Linux, `config-mcp.json` en repli).
- `mcp_python_server.py` : serveur FastMCP maison (natif SDK MCP 2.x) exposant l'exécution de code Python sandboxé ; remplace `mcp-python-interpreter`.
- `config-mcp*.json` : définissent les serveurs MCP dans la clé `mcpServers`.
- `workdir/` : espace de travail du serveur `python` (`scripts/` = fichiers générés par le LLM, `pythonfiles/` = fichiers personnels). L'interpréteur est celui d'uvx (libs via les `--with` de `config-mcp*.json`).
- `test-formulaire/` : harnais de test end-to-end du serveur `playwright` (voir la section dédiée ci-dessous).

## Prérequis

- `uvx` installé (pour lancer `simplexng`, `mcp-proxy`, et le serveur `python` via `mcp_python_server.py`)
- `npx` installé (Node.js) pour lancer `mcp-trunc-proxy`, `wikipedia-mcp`, `mcp-searxng`

## Depannage SearXNG (403)

### 1. Le limiter SearXNG bloque les requetes non-navigateur

Le log peut afficher explicitement : `WARNING  missing config file: limiter.toml`.

Sans ce fichier, SearXNG active une protection anti-bot par defaut qui rejette les requetes HTTP sans User-Agent navigateur (comme celles de `mcp-searxng`).

Creer le fichier `~/.config/simplexng/limiter.toml` pour desactiver ce filtre localement :

```bash
cat > ~/.config/simplexng/limiter.toml << 'EOF'
[botdetection.ip_limit]
link_token = false

[botdetection.ip_lists]
block_ip = []
pass_ip = ["127.0.0.1", "::1"]
EOF
```

Note : `start-mcp.sh` cree ce fichier automatiquement s'il est absent, donc ce warning disparait des les lancements suivants du script.

### 2. Le format JSON doit etre active dans les settings

Dans `~/.config/simplexng/simplexng_settings.yml`, verifier/ajouter :

```yaml
search:
  formats:
    - html
    - json
```

Sans cela, les appels API JSON (comme ceux de `mcp-searxng`) peuvent recevoir un `403`.

## Démarrage

```bash
chmod +x start-mcp.sh
./start-mcp.sh
```

Le script fait, dans cet ordre :

1. Vérifie si SearXNG tourne déjà sur le port 8888 : si oui, réutilise l'instance existante ; sinon, le démarre via `uvx --with sniffio --with anyio simplexng` (en arrière-plan, logs dans `searxng.log`).
2. S'assure de la présence de `limiter.toml` et avertit si le format JSON manque dans `simplexng_settings.yml`.
3. Démarre `mcp-proxy` avec :
   - `--with "mcp<2.0.0"` (voir note ci-dessous)
   - `--named-server-config config-mcp-macos.json` (ou `config-mcp-linux.json` selon l'OS, avec repli sur `config-mcp.json`)
   - `--allow-origin` configuré par défaut (`https://palgania.ovh:8106`, `localhost:8080`, etc.) ou surchargé via la variable d'environnement `MCP_ALLOW_ORIGINS`
   - `--port 8001`
   - `--stateless`
4. À l'arrêt de `mcp-proxy` (ex: `Ctrl+C`), termine le processus SearXNG uniquement si celui-ci a été lancé par le script.

> **Note sur le pin `mcp<2` (proxy uniquement)** : la version 0.12.0 de `mcp-proxy` n'est pas compatible avec le SDK MCP 2.x (qui a supprimé `request_ctx` de `mcp.server.lowlevel.server`). Sans ce pin, `uvx` résout `mcp>=2` et `mcp-proxy` plante à l'import. Le pin `--with "mcp<2.0.0"` reste donc nécessaire dans `start-mcp.sh` pour le proxy. À retirer uniquement quand `mcp-proxy` publiera une version compatible SDK 2.x (aucune à ce jour, y compris la branche `main`).
>
> Le serveur `python`, lui, est désormais natif SDK 2.x : `mcp-python-interpreter` (qui importait `mcp.server.fastmcp`, supprimé en mcp 2.x) a été remplacé par `mcp_python_server.py`, un serveur FastMCP maison (voir ci-dessous), sans pin de version.

## Serveurs exposés

Le fichier `config-mcp*.json` de l'OS expose 4 serveurs dans `mcpServers` :

- `wikipedia`
  - via `wikipedia-mcp`
  - langue forcée en français (`--language fr`)
  - encapsulé avec `mcp-trunc-proxy` pour limiter la taille des réponses
- `search`
  - via `mcp-searxng`
  - variable d'environnement `SEARXNG_URL=http://localhost:8888`
  - encapsulé avec `mcp-trunc-proxy`
- `python`
  - via `mcp_python_server.py` (serveur FastMCP maison, natif SDK MCP 2.x) — remplace `mcp-python-interpreter`
  - outils :
    - `run_python_code(code)` : exécute du code inline dans un subprocess et renvoie stdout+stderr
    - `run_python_file(path)` : exécute un fichier `.py` situé dans le sandbox
    - `list_sandbox_files()` : liste l'ensemble des fichiers et dossiers dans le sandbox
    - `read_sandbox_file(path)` : lit le contenu d'un fichier texte présent dans le sandbox
  - interpréteur de l'env `uvx` avec bibliothèques pré-incluses : `sympy`, `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn`, `scikit-learn`, `requests`
  - garde-fous subprocess : timeout wall-clock (30 s) intercepté proprement, limite CPU (25 s), limite taille fichier écrit (50 MB), limite mémoire 2 GB (Linux seulement — `RLIMIT_AS` n'est pas fiable sur macOS)
  - protection de contexte : tronquage automatique de la sortie si celle-ci dépasse 15 000 caractères
  - accès fichiers confiné à `workdir/scripts/` (sandbox : tout chemin relatif sortant est bloqué)
- `playwright`
  - via `@playwright/mcp` (serveur officiel Playwright MCP)
  - `--browser chrome` : utilise le Chrome/Chromium installé sur la machine (remplacer par `firefox`, `webkit` ou `msedge` si besoin)
  - expose la navigation, le clic, la saisie de texte, le remplissage de formulaires texte + cases à cocher (`browser_fill_form`), la sélection dans les listes déroulantes, la gestion des onglets et des dialogues
  - le navigateur travaille sur des snapshots d'arbre d'accessibilité (pas de captures d'écran), plus fiables pour un LLM
  - mode visible par défaut ; profil persistant conservé entre les sessions (cookies, connexions) ; options `--headless` (invisible) et `--isolated` (profil jetable) à ajouter dans les `args` si souhaité
  - non encapsulé par `mcp-trunc-proxy` (les snapshots contiennent des `ref` utilisées par les appels suivants, il ne faut pas les tronquer)

## URLs à utiliser côté client MCP

Une fois lancé, les points d'accès utiles sont :

- http://127.0.0.1:8001/servers/wikipedia/mcp
- http://127.0.0.1:8001/servers/search/mcp
- http://127.0.0.1:8001/servers/python/mcp
- http://127.0.0.1:8001/servers/playwright/mcp

Important : utiliser les points d'accès en `/mcp` côté client (dans l'interface web de llama-server, rubrique MCP), même si certains logs de `mcp-proxy` affichent aussi des URLs en `/sse`.

On peut aussi faire des recherche web directement à l'adresse `http://127.0.0.1:8888` (SearXNG) pour tester que SearXNG fonctionne correctement.

## Test du serveur `playwright` (remplissage de formulaire)

Un harnais de test end-to-end est fourni dans `test-formulaire/` :

- `form.html` : formulaire local (champs texte, cases à cocher, liste déroulante) qui affiche ses valeurs soumises dans la page
- `driver.py` : pilote le serveur Playwright MCP en JSON-RPC sur stdio, comme le ferait un client MCP : `browser_navigate` -> `browser_snapshot` (refs d'accessibilité) -> `browser_fill_form` -> `browser_click` (Envoyer) -> vérification des valeurs soumises
- `run.sh` : sert `form.html` sur le port 8765, lance le driver, puis arrête le serveur HTTP

```bash
cd test-formulaire
chmod +x run.sh
./run.sh
```

Le test réussit si la sortie se termine par `TEST REUSSI` (valeurs attendues : `Jean Dupont`, `jean.dupont@example.com`, newsletter et CGU cochées, taille `M`). Prérequis : `npx` (Node.js) et Chrome installé. Le navigateur est lancé en mode `--headless --isolated` (invisible, sans impact sur le profil de navigation).

> **Note sur le schéma des outils** : la version `@latest` de `@playwright/mcp` exige, pour `browser_fill_form`, des champs de la forme `{target, name, type, value}` (`type` parmi `textbox`, `checkbox`, `radio`, `combobox`, `slider` ; `value` `true`/`false` pour une case à cocher) et, pour `browser_click`, l'argument `target` (et non `ref`).

## Lancement de llama-server avec MCP

Ordre recommande :

1. Lancer d'abord les serveurs MCP :

```bash
./start-mcp.sh
```

2. Lancer ensuite `llama-server` avec l'interface web MCP active (ajuster selon les dossiers choisis pour stocker les modèles) :

```bash
./build/bin/llama-server \
  -m ../models_llm/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --mmproj ../models_llm/Qwen3.5-35B-A3B/mmproj-F16.gguf \
  -c 262144 \
  --chat-template-kwargs '{"enable_thinking": true}' \
  -ctk q4_0 \
  -ctv q4_0 \
  --jinja \
  --webui-mcp-proxy
```

Dans l'interface web de `llama-server`, ajouter les points d'acces MCP en `/mcp` listes dans la section precedente.
