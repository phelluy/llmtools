---
name: voxtral3-cr
description: Transcrire et diariser un enregistrement audio/video (BigBlueButton, reunion, cours, entretien) avec l'API Mistral Voxtral 3, puis produire un compte rendu. A charger quand l'utilisateur demande de transcrire un enregistrement, de telecharger une video BBB, d'identifier qui parle, ou de produire un compte rendu de reunion.
---

# Voxtral 3 — Transcription, diarisation, compte rendu

Pipeline valide en production (octobre 2026) sur des enregistrements BigBlueButton
Scalelite. Le script `voxtral3-cr.sh` de ce dossier fait tout le travail pour le cas
synchrone (audio < ~30 min) : l'utiliser plutot que de reinventer les appels curl.

## Usage rapide

```bash
./voxtral3-cr.sh \
  -n "speaker_1=<Nom Locuteur 1>,speaker_2=<Nom Locuteur 2>" \
  -c "<MotMalTranscrit>=<Correction>" \
  -r \
  "https://<serveur-bbb>/playback/video/<id>/"
```

Sorties : `<base>-transcription.txt` (horodate, locuteurs nommes),
`<base>-compte-rendu.md` (avec `-r`), `<base>.transcription.json` (reponse brute).

## Etapes du pipeline

1. **Recuperation** — enregistrement BBB : l'URL de playback
   `.../playback/video/<id>/` sert directement le fichier :
   remplacer par `.../video/<id>/video-0.m4v` (telechargeable sans authentification).
   Pour les metadonnees (participants, duree) :
   `.../presentation/<id>/metadata.xml`.
   Limite : ne gere que le format playback "video". Pour le format
   "presentation" (lecteur diapos + webcam), inspecter la page de playback
   pour trouver l'URL du fichier video avant d'utiliser le script, ou
   fournir directement un fichier local au script.
2. **Extraction audio** — `ffmpeg -i video.mp4 -vn -c:a copy audio.m4a`
   (copie sans reencodage, ~40 Mo video -> ~6 Mo audio pour 6 min).
3. **Upload** — `POST /v1/files` avec `purpose=audio` -> `file_id`.
   L'upload est separe du traitement : on peut retraiter le meme file_id
   N fois sans renvoyer le fichier.
4. **Transcription** — `POST /v1/audio/transcriptions` avec `file_id`.
5. **Post-traitement** — nommage des locuteurs, corrections de vocabulaire,
   mise en forme. Le script appelle `postprocess.py`.
6. **Compte rendu** (option `-r`) — `mistral-small-latest` en chat completions
   sur la transcription finale.

## Parametres de l'API transcription (valides empiriquement)

- `model` : `voxtral-small-transcribe-3` (meilleure qualite, $0,004/min)
  ou `voxtral-mini-transcribe-3` ($0,0035/min).
- Entree audio : `file` (multipart direct), `file_url` (URL publique) ou
  `file_id` (fichier deja uploade).
- `diarize=true` : **exige** `timestamp_granularities=segment` en meme temps,
  sinon erreur 422. Les segments renvoient `speaker_id` (`speaker_1`, ...).
- `language=fr` : accepte mais le champ revient vide dans la reponse
  (la detection auto fonctionne bien en francais).
- `prompt` et `keyphrases` : acceptes, effet reel faible (voir limites).

## Limites connues (ne pas perdre de temps a retester)

- **La diarisation ne nomme pas** : `speaker_id` reste generique meme si le
  prompt donne les noms. Le mapping vers les vrais noms se fait en
  post-traitement (`-n`), a partir du contexte (qui mene la reunion, timbres).
- Le prompt ne corrige pas les prenoms phonetiquement ambigus :
  testes `prompt` texte libre, `keyphrases` en JSON et en champs repetes —
  un prenom reste transcrit avec la graphie la plus frequente du son entendu.  Prevoir des corrections post-traitement (`-c "<Faux>=<Vrai>"`).
- Les messages de chat d'un enregistrement BBB ne sont pas dans l'audio :
  ils sont stockes separement (non transcrits par Voxtral). Si le contexte
  du chat importe, les recuperer dans les fichiers de l'enregistrement.
- Jusqu'a 3 h d'audio en une seule requete selon la doc, mais au-dela de
  ~30 min, preferer le mode batch ci-dessous.

## Mode batch asynchrone (audio long, > ~30 min)

Valide empiriquement : le Batch API accepte l'endpoint `/v1/audio/transcriptions`.

1. Upload audio : `POST /v1/files` `purpose=audio` -> file_id.
2. URL signee : `GET /v1/files/{file_id}/url`.
3. JSONL d'une ligne par audio : `{"body": {"model": "...", "file_url": "<url signee>",
   "diarize": true, "timestamp_granularities": ["segment"]}}`
   puis `POST /v1/files` avec `purpose=batch` -> jsonl_id.
4. `POST /v1/batch/jobs` avec
   `{"input_files": ["<jsonl_id>"], "endpoint": "/v1/audio/transcriptions",
   "model": "voxtral-small-transcribe-3"}` -> job_id.
5. Poll : `GET /v1/batch/jobs/{job_id}` (statuts QUEUED, RUNNING, SUCCESS).
   SLA 24 h, mais observe ~40 s sur un job d'un fichier.
6. Resultat : `GET /v1/files/{output_file}/content` -> JSONL, la transcription
   est dans `response.body` (memes champs que l'appel synchrone).

La remise batch de 50% (chat/embeddings/OCR) n'est pas confirmee pour le
speech facture a la minute ; ne pas la promettre a l'utilisateur.

## Ordres de grandeur

- Traitement : ~4 s par minute d'audio (6 min -> ~25 s ; 2 h -> ~8-15 min).
- Cout : $0,004/min (small) — 6 min ≈ 2,6 centimes ; 2 h ≈ $0,48.
- Les essais de parametres coutent chacun un appel complet : reutiliser le
  `file_id` plutot que de renvoyer le fichier.
