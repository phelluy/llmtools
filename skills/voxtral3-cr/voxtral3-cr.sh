#!/usr/bin/env bash
# voxtral3-cr.sh — Transcription + diarisation Voxtral 3 (API Mistral) + compte rendu optionnel
#
# Usage:
#   ./voxtral3-cr.sh [options] <fichier audio/video | URL BBB>
#
# Options:
#   -m, --model <id>        voxtral-small-transcribe-3 (defaut) ou voxtral-mini-transcribe-3
#   -l, --lang <code>       fr (defaut)
#   -n, --names <map>       "speaker_1=Philippe Helluy,speaker_2=Line El Kamel"
#   -c, --corrections <map> remplacements apres transcription, ex: "Oriane=Aurian"
#   -r, --resume            genere aussi un compte rendu (chat mistral-small-latest)
#   -o, --output <fichier>  sortie (defaut: <base>-transcription.txt)
#   -h, --help
#
# Prerequis: MISTRAL_API_KEY dans l'environnement, ffmpeg dans le PATH.

set -euo pipefail

MODEL="voxtral-small-transcribe-3"
LANGUE="fr"
NAMES=""
CORRECTIONS=""
RESUME=0
OUTPUT=""
INPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="$2"; shift 2 ;;
    -l|--lang) LANGUE="$2"; shift 2 ;;
    -n|--names) NAMES="$2"; shift 2 ;;
    -c|--corrections) CORRECTIONS="$2"; shift 2 ;;
    -r|--resume) RESUME=1; shift ;;
    -o|--output) OUTPUT="$2"; shift 2 ;;
    -h|--help) grep '^# ' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) INPUT="$1"; shift ;;
  esac
done

[[ -z "$INPUT" ]] && { echo "Erreur: fichier ou URL manquant. -h pour l'aide." >&2; exit 1; }
[[ -z "${MISTRAL_API_KEY:-}" ]] && { echo "Erreur: MISTRAL_API_KEY non defini." >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "Erreur: ffmpeg absent." >&2; exit 1; }

API="https://api.mistral.ai"
BASE="$(basename "$INPUT" | sed 's/\.[^.]*$//')"
DIR="$(pwd)"
AUDIO=""

cleanup() { [[ -n "$AUDIO" && -f "$AUDIO" ]] && rm -f "$AUDIO"; }
trap cleanup EXIT

# --- 1. Recuperation de l'entree ---
if [[ "$INPUT" =~ ^https?:// ]]; then
  # Enregistrement BigBlueButton (Scalelite/numerique-esr):
  # /playback/video/<id>/  ->  /video/<id>/video-0.m4v
  if [[ "$INPUT" == *"/playback/video/"* ]]; then
    VIDEO_URL="${INPUT/\/playback\/video\///video/}video-0.m4v"
  elif [[ "$INPUT" == */playback/* ]]; then
    echo "Erreur: format de playback non gere (attendu /playback/video/<id>/)." >&2; exit 1
  else
    VIDEO_URL="$INPUT"
  fi
  echo "1/5 Telechargement video BBB..." >&2
  curl -sS -f -o "$BASE.tmp.video" "$VIDEO_URL" --max-time 600
  INPUT="$BASE.tmp.video"
  trap 'rm -f "$BASE.tmp.video"' EXIT
fi

# --- 2. Extraction audio ---
echo "2/5 Extraction audio (ffmpeg)..." >&2
AUDIO="${INPUT%.*}.tmp-audio.m4a"
ffmpeg -y -v error -i "$INPUT" -vn -c:a copy "$AUDIO" 2>/dev/null || \
  ffmpeg -y -v error -i "$INPUT" -vn -c:a aac -b:a 128k "$AUDIO"

# --- 3. Upload vers Files API (une fois, retentable independamment) ---
echo "3/5 Upload vers Mistral Files API..." >&2
FILE_ID=$(curl -sS -f "$API/v1/files" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F purpose="audio" -F "file=@$AUDIO" \
  --max-time 600 | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")

# --- 4. Transcription + diarisation (synchrone; voir SKILL.md pour le mode batch >30 min) ---
echo "4/5 Transcription ($MODEL, diarisation)... audio ~4 s/min de traitement" >&2
curl -sS -f "$API/v1/audio/transcriptions" \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -F model="$MODEL" \
  -F file_id="$FILE_ID" \
  -F diarize=true \
  -F timestamp_granularities="segment" \
  -F language="$LANGUE" \
  --max-time 1800 -o "${BASE}.transcription.json"

# --- 5. Mise en forme (noms + corrections) ---
echo "5/5 Mise en forme..." >&2
[[ -z "$OUTPUT" ]] && OUTPUT="${BASE}-transcription.txt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/postprocess.py" "${BASE}.transcription.json" "$OUTPUT" "$NAMES" "$CORRECTIONS" "$BASE"

# --- 6. Compte rendu optionnel ---
if [[ "$RESUME" -eq 1 ]]; then
  echo "Bonus: compte rendu (mistral-small-latest)..." >&2
  python3 - "$OUTPUT" "${BASE}-compte-rendu.md" <<'PYEOF'
import json, os, urllib.request
api = "https://api.mistral.ai"
key = os.environ["MISTRAL_API_KEY"]
transcript = open(os.argv[1], encoding="utf-8").read()
body = {
    "model": "mistral-small-latest",
    "messages": [
        {"role": "user", "content":
            "Redige un compte rendu structuré en français (décisions, points clés, "
            "chronologie) de la transcription de réunion suivante. Utilise les vrais "
            "noms des locuteurs. Format Markdown avec sections.\n\n" + transcript},
    ],
}
req = urllib.request.Request(
    api + "/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req, timeout=300))
open(os.argv[2], "w", encoding="utf-8").write(resp["choices"][0]["message"]["content"])
print("Compte rendu:", os.argv[2])
PYEOF
fi

echo "OK: $OUTPUT" >&2
