# mcp-configs

## Comment ça marche avec mcp-trunc-proxy

Chaque serveur MCP (Wikipedia, StackOverflow, Search) est lancé derrière `mcp-trunc-proxy`.

Le principe est le suivant :

1. Le client appelle `mcp-proxy` sur le port `8001`.
2. `mcp-proxy` route la requête vers le serveur nommé configuré dans `config-mcp.json`.
3. Ce serveur nommé passe d'abord par `mcp-trunc-proxy`, qui limite la taille des réponses (`max-bytes`, `preview-max-chars`, etc.).
4. `mcp-trunc-proxy` appelle ensuite le serveur MCP réel (`wikipedia-mcp`, `stackoverflow-mcp` ou `mcp-searxng`).

Ce montage évite les réponses trop longues, réduit le bruit dans les sorties et garde des résultats plus stables côté client.

Configuration locale de serveurs MCP (Wikipedia, StackOverflow, Search via SearXNG) exposés par `mcp-proxy` sur le port `8001`.

## Fichiers du dépôt

- `start-mcp.sh` : démarre `simplexng` en arrière-plan, puis lance `mcp-proxy` avec `config-mcp.json`.
- `config-mcp.json` : définit les serveurs MCP dans la clé `mcpServers`.

## Prérequis

- `uvx` installé (pour lancer `simplexng`, `mcp-proxy`, `stackoverflow-mcp`)
- `npx` installé (Node.js) pour lancer `mcp-trunc-proxy`, `wikipedia-mcp`, `mcp-searxng`

## Démarrage

```bash
chmod +x start-mcp.sh
./start-mcp.sh
```

Le script fait, dans cet ordre :

1. Démarre SearXNG via `uvx --with sniffio --with anyio simplexng` (en arrière-plan).
2. Attend 3 secondes.
3. Démarre `mcp-proxy` avec :
   - `--named-server-config config-mcp.json`
   - `--allow-origin "*"`
   - `--port 8001`
   - `--stateless`
4. À l'arrêt de `mcp-proxy` (ex: `Ctrl+C`), termine le processus SearXNG lancé par le script.

## Serveurs exposés

Le fichier `config-mcp.json` expose 3 serveurs dans `mcpServers` :

- `wikipedia`
  - via `wikipedia-mcp`
  - langue forcée en français (`--language fr`)
  - encapsulé avec `mcp-trunc-proxy` pour limiter la taille des réponses
- `stackoverflow`
  - via `stackoverflow-mcp`
  - encapsulé avec `mcp-trunc-proxy`
- `search`
  - via `mcp-searxng`
  - variable d'environnement `SEARXNG_URL=http://localhost:8888`
  - encapsulé avec `mcp-trunc-proxy`

## URLs à utiliser côté client MCP

Une fois lancé, les points d'accès utiles sont :

- http://127.0.0.1:8001/servers/wikipedia/mcp
- http://127.0.0.1:8001/servers/stackoverflow/mcp
- http://127.0.0.1:8001/servers/search/mcp

Important : utiliser les points d'accès en `/mcp` côté client (dans l'interface web de llama-server, rubrique MCP), même si certains logs de `mcp-proxy` affichent aussi des URLs en `/sse`.

On peut aussi faire des recherche web directement à l'adresse `http://127.0.0.1:8888` (SearXNG) pour tester que SearXNG fonctionne correctement.
