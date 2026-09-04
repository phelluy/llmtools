#!/bin/bash
set -e

# Se placer dans le répertoire du script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Choisir la config selon l'OS (avec repli sur config-mcp.json)
case "$(uname)" in
  Linux)
    CONFIG_FILE="$SCRIPT_DIR/config-mcp-linux.json"
    ;;
  Darwin)
    CONFIG_FILE="$SCRIPT_DIR/config-mcp-macos.json"
    ;;
  *)
    CONFIG_FILE="$SCRIPT_DIR/config-mcp.json"
    ;;
esac

# Repli sur config-mcp.json si la config spécifique n'existe pas
if [[ ! -f "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$SCRIPT_DIR/config-mcp.json"
fi

# Vérifier que la config existe
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Erreur: fichier de config introuvable: $CONFIG_FILE" >&2
  exit 1
fi

# Cleanup au signal (n'arrête SearXNG que s'il a été lancé par ce script)
SEARXNG_PID=""
cleanup() {
  if [[ -n "$SEARXNG_PID" ]]; then
    echo "Arrêt de SearXNG (PID: $SEARXNG_PID)..."
    kill "$SEARXNG_PID" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup EXIT INT TERM

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

# Vérifier que le format JSON est activé dans simplexng_settings.yml (évite les erreurs 403)
SETTINGS_FILE="$HOME/.config/simplexng/simplexng_settings.yml"
if [[ -f "$SETTINGS_FILE" ]] && ! grep -q -- "- json" "$SETTINGS_FILE"; then
  echo "Attention: le format JSON semble manquant dans $SETTINGS_FILE."
  echo "Vérifiez que 'formats:' contient bien '- json' pour autoriser les requêtes de mcp-searxng."
fi

# Préparer le workdir du serveur python : sandbox où s'exécutent les fichiers
# générés par le LLM. L'interpréteur est celui d'uvx (configuré dans
# config-mcp*.json via les --with) ; plus de venv séparé.
mkdir -p "$SCRIPT_DIR/workdir/scripts"

# 1. Vérifier si SearXNG tourne déjà sur le port 8888 ou le lancer
if curl -s --connect-timeout 1 http://localhost:8888 > /dev/null 2>&1; then
  echo "SearXNG est déjà en cours d'exécution sur le port 8888 (réutilisation de l'instance existante)."
else
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
fi

# 2. Lancer mcp-proxy
# NB: mcp-proxy 0.12.0 ne supporte pas encore le SDK mcp 2.x (request_ctx supprimé),
# on épingle mcp<2.0.0 pour que uvx résolve une version compatible.

# Vérifier que le port 8001 est libre avant de lancer mcp-proxy
if curl -s --connect-timeout 1 http://127.0.0.1:8001 > /dev/null 2>&1; then
  echo "Erreur: le port 8001 est déjà utilisé. mcp-proxy est-il déjà en cours d'exécution ?" >&2
  exit 1
fi

# Origines autorisées : surchargeable via la variable d'environnement MCP_ALLOW_ORIGINS
# Doit correspondre exactement au schéma + hôte + port avec lesquels llama-server
# est accédé dans le navigateur.
DEFAULT_ORIGINS=(
  "https://palgania.ovh:8106"
  "http://localhost:8080"
  "http://127.0.0.1:8080"
  "http://localhost:6806"
)

if [[ -n "$MCP_ALLOW_ORIGINS" ]]; then
  # Découper la chaîne en tableau d'arguments
  read -r -a ORIGINS <<< "$MCP_ALLOW_ORIGINS"
else
  ORIGINS=("${DEFAULT_ORIGINS[@]}")
fi

echo "Démarrage de mcp-proxy..."
uvx --with "mcp<2.0.0" mcp-proxy \
  --named-server-config "$CONFIG_FILE" \
  --allow-origin "${ORIGINS[@]}" \
  --port 8001 \
  --stateless
