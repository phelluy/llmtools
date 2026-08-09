# mcp-configs

## Comment ça marche avec mcp-trunc-proxy

Les serveurs MCP `wikipedia` et `search` sont lancés derrière `mcp-trunc-proxy`.

Le principe est le suivant :

1. Le client appelle `mcp-proxy` sur le port `8001`.
2. `mcp-proxy` route la requête vers le serveur nommé configuré dans le fichier `config-mcp*.json` de l'OS.
3. Ce serveur nommé passe d'abord par `mcp-trunc-proxy`, qui limite la taille des réponses (`max-bytes`, `preview-max-chars`, etc.).
4. `mcp-trunc-proxy` appelle ensuite le serveur MCP réel (`wikipedia-mcp` ou `mcp-searxng`).

Ce montage évite les réponses trop longues, réduit le bruit dans les sorties et garde des résultats plus stables côté client.

Configuration locale de serveurs MCP (Wikipedia, Search via SearXNG, et Python) exposés par `mcp-proxy` sur le port `8001`.

## Fichiers du dépôt

- `start-mcp.sh` : démarre `simplexng` en arrière-plan, puis lance `mcp-proxy` avec la config adaptée à l'OS (`config-mcp-macos.json` sur macOS, `config-mcp-linux.json` sur Linux, `config-mcp.json` en repli).
- `config-mcp*.json` : définissent les serveurs MCP dans la clé `mcpServers`.
- `workdir/` : espace de travail du serveur `python` (`scripts/` = fichiers générés par le LLM, `pythonfiles/` = fichiers personnels, `.venv/` = interpréteur, `requirements.txt` = dépendances du venv).

## Prérequis

- `uvx` installé (pour lancer `simplexng`, `mcp-proxy`, `mcp-python-interpreter`)
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

1. Démarre SearXNG via `uvx --with sniffio --with anyio simplexng` (en arrière-plan, logs dans `searxng.log`).
2. Attend que le port 8888 réponde (jusqu'à 15 secondes).
3. Démarre `mcp-proxy` avec :
   - `--with "mcp<2.0.0"` (voir note ci-dessous)
   - `--named-server-config config-mcp-macos.json` (ou `config-mcp-linux.json` selon l'OS)
   - `--allow-origin "https://palgania.ovh:8106" "http://localhost:8080" "http://127.0.0.1:8080"` (accès restreint à llama-server ; le schéma + hôte + port doivent correspondre exactement à l'URL avec laquelle llama-server est accédé dans le navigateur — ex. HTTPS si accès distant)
   - `--port 8001`
   - `--stateless`
4. À l'arrêt de `mcp-proxy` (ex: `Ctrl+C`), termine le processus SearXNG lancé par le script.

> **Note sur `--with "mcp<2.0.0"`** : la version 0.12.0 de `mcp-proxy` n'est pas compatible avec le SDK MCP 2.x (qui a supprimé `request_ctx` de `mcp.server.lowlevel.server`). Sans ce pin, `uvx` résout `mcp>=2` et `mcp-proxy` plante à l'import. À retirer uniquement quand une version de `mcp-proxy` compatible SDK 2.x sera publiée.

## Serveurs exposés

Le fichier `config-mcp*.json` de l'OS expose 3 serveurs dans `mcpServers` :

- `wikipedia`
  - via `wikipedia-mcp`
  - langue forcée en français (`--language fr`)
  - encapsulé avec `mcp-trunc-proxy` pour limiter la taille des réponses
- `search`
  - via `mcp-searxng`
  - variable d'environnement `SEARXNG_URL=http://localhost:8888`
  - encapsulé avec `mcp-trunc-proxy`
- `python`
  - via `mcp-python-interpreter`
  - code inline (`run_python_code`) exécuté dans le processus du serveur, avec les bibliothèques des `--with` (`sympy`, `numpy`, `scipy`, `matplotlib`, `pandas`) ; le venv `workdir/.venv` sert au mode fichier/subprocess (`run_python_file`)
  - accès fichiers confiné à `workdir/scripts/` (sandbox : tout chemin hors de ce dossier est refusé)
  - pour recréer le venv, voir `workdir/requirements.txt`

## URLs à utiliser côté client MCP

Une fois lancé, les points d'accès utiles sont :

- http://127.0.0.1:8001/servers/wikipedia/mcp
- http://127.0.0.1:8001/servers/search/mcp
- http://127.0.0.1:8001/servers/python/mcp

Important : utiliser les points d'accès en `/mcp` côté client (dans l'interface web de llama-server, rubrique MCP), même si certains logs de `mcp-proxy` affichent aussi des URLs en `/sse`.

On peut aussi faire des recherche web directement à l'adresse `http://127.0.0.1:8888` (SearXNG) pour tester que SearXNG fonctionne correctement.

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
