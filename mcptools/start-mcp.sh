#!/bin/bash
set -e

# Se placer dans le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Choisir la config selon l'OS
case "$(uname)" in
  Linux)
    CONFIG_FILE="$SCRIPT_DIR/config-mcp-linux.json"
    ;;
  Darwin)
    CONFIG_FILE="$SCRIPT_DIR/config-mcp-macos.json"
    ;;
  *)
    echo "OS inconnu: $(uname). Utilise config-mcp.json."
    CONFIG_FILE="$SCRIPT_DIR/config-mcp.json"
    ;;
esac

# Vérifier que la config existe
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Erreur: fichier de config introuvable: $CONFIG_FILE" >&2
  exit 1
fi

# Cleanup au signal
cleanup() {
  echo "Arrêt de SearXNG..."
  kill "$SEARXNG_PID" 2>/dev/null || true
  exit 0
}
trap cleanup EXIT INT TERM

# Vérifier que le port 8888 est libre avant de lancer SearXNG
if curl -s --connect-timeout 1 http://localhost:8888 > /dev/null 2>&1; then
  echo "Erreur: le port 8888 est déjà utilisé. SearXNG est-il déjà en cours d'exécution ?" >&2
  exit 1
fi

# S'assurer que limiter.toml existe : sans lui, SearXNG bloque les requêtes
# non-navigateur (comme celles de mcp-searxng) avec un 403.
LIMITER_FILE="$HOME/.config/simplexng/limiter.toml"
if [[ ! -f "$LIMITER_FILE" ]]; then
  mkdir -p "$(dirname "$LIMITER_FILE")"
  cat > "$LIMITER_FILE" << 'EOF'
[botdetection.ip_limit]
link_token = false

[botdetection.ip_lists]
block_ip = []
pass_ip = ["127.0.0.1", "::1"]
EOF
  echo "Fichier $LIMITER_FILE créé (désactive le filtre anti-bot local de SearXNG)."
fi

# 1. Lancer SearXNG en arrière-plan (logs dans searxng.log)
echo "Démarrage de SearXNG..."
uvx --with sniffio --with anyio simplexng > "$SCRIPT_DIR/searxng.log" 2>&1 &
SEARXNG_PID=$!

# Attendre que le port 8888 réponde (plus fiable que sleep fixe)
echo "Attente de SearXNG sur le port 8888..."
for i in $(seq 1 15); do
  curl -s --connect-timeout 1 http://localhost:8888 > /dev/null && break
  sleep 1
done
echo "SearXNG prêt."

# 2. Lancer mcp-proxy
# NB: mcp-proxy 0.12.0 ne supporte pas encore le SDK mcp 2.x (request_ctx supprimé),
# on épingle mcp<2.0.0 pour que uvx résolve une version compatible.

# Vérifier que le port 8001 est libre avant de lancer mcp-proxy
if curl -s --connect-timeout 1 http://127.0.0.1:8001 > /dev/null 2>&1; then
  echo "Erreur: le port 8001 est déjà utilisé. mcp-proxy est-il déjà en cours d'exécution ?" >&2
  exit 1
fi

echo "Démarrage de mcp-proxy..."
# Origines autorisées : doit correspondre exactement au schéma + hôte + port
# avec lesquels llama-server est accédé dans le navigateur (ici HTTPS via
# palgania.ovh:8106, plus localhost:8080 pour un usage local).
uvx --with "mcp<2.0.0" mcp-proxy \
  --named-server-config "$CONFIG_FILE" \
  --allow-origin "https://palgania.ovh:8106" "http://localhost:8080" "http://127.0.0.1:8080" \
  --port 8001 \
  --stateless
