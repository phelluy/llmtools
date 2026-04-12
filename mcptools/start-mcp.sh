#!/bin/bash

# 1. Lancer SearXNG en arrière-plan
echo "Démarrage de SearXNG..."
uvx --with sniffio --with anyio simplexng &
SEARXNG_PID=$!

# Attendre que le port 8888 réponde (plus fiable que sleep fixe)
echo "Attente de SearXNG sur le port 8888..."
for i in $(seq 1 15); do
  curl -s http://localhost:8888 > /dev/null && break
  sleep 1
done
echo "SearXNG prêt."

# 2. Lancer mcp-proxy
echo "Démarrage de mcp-proxy..."
uvx mcp-proxy \
  --named-server-config config-mcp.json \
  --allow-origin "*" \
  --port 8001 \
  --stateless

# 3. Arrêter SearXNG proprement quand mcp-proxy s'arrête (Ctrl+C)
echo "Arrêt de SearXNG..."
kill $SEARXNG_PID