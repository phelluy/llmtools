#!/usr/bin/env python3
"""Post-traitement d'une reponse /v1/audio/transcriptions :
noms des locuteurs, corrections de vocabulaire, format [mm:ss] Nom: texte."""
import json
import sys


def parse_map(s):
    """'a=1,b=2' -> {'a': '1', 'b': '2'} (tolere les virgules dans les valeurs via '=')"""
    out = {}
    if not s:
        return out
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def ts(seconds):
    m, sec = divmod(int(seconds), 60)
    return f"{m:02d}:{sec:02d}"


def main():
    src, dst, names_s, corr_s, base = sys.argv[1:6]
    with open(src, encoding="utf-8") as f:
        d = json.load(f)
    names = parse_map(names_s)
    corrections = parse_map(corr_s)
    segments = d.get("segments", [])
    known_speakers = sorted({s["speaker_id"] for s in segments if s.get("speaker_id")})

    lines = [f"# Transcription - {base}", f"# Modele: {d.get('model')}, diarisation activee", ""]
    for seg in segments:
        text = seg["text"].strip()
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        speaker = seg.get("speaker_id")
        label = names.get(speaker, speaker or "?")
        lines.append(f"[{ts(seg['start'])}] {label}: {text}")

    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    unnamed = [s for s in known_speakers if s not in names]
    if unnamed:
        print(f"Locuteurs sans nom fourni: {', '.join(unnamed)}", file=sys.stderr)


if __name__ == "__main__":
    main()
