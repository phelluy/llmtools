---
name: thunderbird
description: "Retrouver, lire et écrire des courriels dans les boîtes Thunderbird (profils snap ou classiques, mbox locaux) en lecture seule, ou poser des brouillons dans un dossier local sans rien envoyer. À charger dès qu'une tâche touche aux messages ou au profil Thunderbird : chercher un courriel, extraire des pièces jointes, préparer des réponses en brouillon."
---

# Thunderbird : boîtes et profils locaux

## Sûretés (non négociables)

- **Lecture seule par défaut**. Thunderbird peut être ouvert ou fermé pour lire.
- **Jamais l'Outbox** (« Messages en attente ») : Thunderbird propose d'envoyer
  son contenu dès qu'il repasse en ligne. C'est le seul dossier à éviter.
- Ne **jamais envoyer**, ne jamais déplacer ni supprimer un message existant,
  ne jamais toucher aux drapeaux (`X-Mozilla-Status`) d'un message déjà reçu.
- **Écrire seulement Thunderbird fermé** : le demander explicitement à
  l'utilisateur, puis vérifier `pgrep -x thunderbird-bin` (jamais
  `pgrep -f thunderbird`, qui matcherait la ligne de commande du shell).
  `drafts.py --install` impose les deux : drapeau `--tb-stopped` exigé, et
  refus si `thunderbird-bin` est trouvé (le doute vaut refus).
- **Ajouter** un fichier, n'en modifier aucun ; sauvegarder avant toute
  écriture ; vérifier après coup que les mbox et `.msf` existants ont la même
  taille et la même date qu'avant.
- Ne **jamais écrire dans le cache d'un dossier IMAP** : un message ajouté à la
  main y disparaît à la resynchronisation. Passer par un dossier local, puis
  laisser Thunderbird faire l'APPEND (copier-coller de l'utilisateur).

## Retrouver et lire des courriels

```bash
python3 scripts/collect.py --since "2026-10-01 00:00" --until "2026-10-08 08:00" \
        --from-regex '@etu\.' --list-only
```

Lit les magasins locaux (profils snap, classiques et macOS, découverts via
`prefs.js`), en lecture seule, en remontant chaque mbox depuis la fin — les
gros fichiers ne sont jamais lus en entier. `--list-only` d'abord, puis
`--extract` quand la fenêtre est bonne, `--boxes <fichier mbox>` pour cibler.
Sortie : `index_rendus.json` (expéditeur, date, sujet, `message-id`, corps,
pièces jointes renommées `NOM_Prenom__nom_original`).

Pièges : sujets en RFC 2047 (`=?utf-8?Q?...?=`), noms de pièces en cp1252,
messages « Re: » qui ne sont pas des réponses (vérifier date, expéditeur,
`Message-ID`).

## Poser des brouillons

```bash
python3 scripts/drafts.py --dir reponses --attach corrige.pdf \
        --install "Brouillons exo11" --tb-stopped
```

Construit des `.eml` complets (corps en `text/plain; format=flowed` :
paragraphes sur une ligne logique, retours mous, **jamais** de retour dur à 72
caractères) et les pose dans un **nouveau fichier** de `Mail/Local Folders/`,
jamais dans un fichier existant, jamais dans l'Outbox. Thunderbird fermé.
L'utilisateur sélectionne tout et recopie vers le dossier **Brouillons** du
compte : c'est ce copier-coller qui fait l'APPEND serveur.

## Formats, profils, plan B

Voir `references/thunderbird.md` : découverte des profils (snap et classique),
clés `prefs.js`, formats mbox locaux vs IMAP, `X-Mozilla-Status`, vérification
d'un livrage, plan B (glisser les `.eml` dans le dossier visé).

## Frontières

- Corriger des copies d'étudiants : skill `copies-mail`.
- Agendas (calendriers CalDAV, cache `cache.sqlite`) : skill `agenda-mail`.
