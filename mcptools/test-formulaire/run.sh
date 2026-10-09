#!/bin/bash
# Test end-to-end du serveur MCP playwright : sert form.html en local,
# pilote le serveur Playwright MCP via driver.py, puis arrête le serveur HTTP.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if curl -s --connect-timeout 1 http://127.0.0.1:8765 > /dev/null 2>&1; then
  echo "Erreur : le port 8765 est déjà utilisé (un http.server tourne déjà ?)." >&2
  exit 1
fi

# Servir le formulaire de test sur le port 8765
python3 -m http.server 8765 > /dev/null 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
sleep 1

# Le premier lancement peut télécharger le paquet npm (~1 min) ;
# les suivants démarrent en quelques secondes.
timeout 120 python3 driver.py
