---
name: sumup-big-pdf
description: Charger quand l'utilisateur demande de résumer ou compresser un gros document PDF (thèse, HDR, rapport, manuscrit, livre — typiquement 50 à 500 pages) en un fichier Markdown compact de type resume.md, destiné à être ingéré par un LLM et non lu par un humain. Le skill couvre l'OCR du PDF, le découpage en sections vérifié, puis la lecture complète et la compression télégraphique de chaque section, tour à tour, sans jamais sauter de texte.
---

# sumup-big-pdf : compression d'un gros PDF en resume.md pour LLM

## Objectif et contrat

Produire `resume.md` : une **compression** du document qui conserve les éléments
essentiels — objectifs, définitions, formules (LaTeX conservé), énoncés des
théorèmes/ propositions, idées des preuves (squelette d'argument, pas les
détails calculatoires), résultats avec leurs hypothèses, résultats numériques
(setups + ordres observés + conclusions). Le style est **télégraphique** :
dense, sans redondance ni blabla, optimisé pour un LLM. Ce n'est PAS un résumé
littéraire.

**Règle absolue : tout le texte du document doit être lu au moins une fois.**
Aucun chapitre, annexe ou bibliographie ne peut être sauté sous prétexte qu'il
est technique ou répétitif. En revanche, la *compression* doit fusionner les
redondances (ex : résumé EN ignoré si le FR est gardé, chapitre récapitulatif
remplacé par « cf. sections concernées »).

## Paramètres par défaut (à ne pas re-demander systématiquement)

- Langue du résumé = langue du document.
- Le résumé est écrit dans le dossier de travail : `<dossier>/resume.md`.
- Bouts intermédiaires dans `<dossier>/chunks/`.
- Bibliographie : « très compacte » — lister seulement les références clés
  effectivement citées dans le résumé, une ligne chacune, avec le total lu.
  (Si l'utilisateur exprime une préférence différente au moment de la demande,
  la suivre.)

## Algorithme

### Phase 0 — État et attente

- Créer (`todo`) une tâche par phase, une tâche par bout.
- Écrire dans le **scratchpad** un fichier d'état (liste des bouts, ordre de
  traitement, progression, contrat). Le maintenir à jour **après chaque bout** :
  c'est ce qui permet de reprendre si la compaction du contexte de Vibe se
  déclenche en cours de route (relire l'état + le bout en cours, et reprendre —
  `resume.md` étant sur disque, rien n'est perdu).
- L'OCR et le découpage produisent des fichiers sur disque : tout doit être
  rejouable après compaction du contexte.

### Phase 1 — OCR (charger le skill `pdf2md`)

1. Charger le skill `pdf2md`, créer le venv : `uv venv && uv pip install pymupdf requests`.
2. OCR via `mistral-pdf2md.py` sur le **dossier** du PDF (pas le fichier) :
   - Doit tourner en **arrière-plan** via `tools.process.start` (run_typescript),
     JAMAIS avec `&` dans le sandbox `bash` (le fils est tué au timeout du shell).
   - `mistral-pdf2md.py` traite tous les PDF du dossier, ignore les `.md`
   existants et les dossiers cachés (`.venv` inclus) ; pas de reprise → si
   interruption, relancer (le `.md` existant est sauté). Alternative : le
   script UNISTRA est par contre résumable page par page.
   - Attendre la fin par le mécanisme d'attente passive (notification terminal du
     runtime / `wait`), pas par boucle de polling (budget sandbox ≈ 60 s par
     programme run_typescript).
3. Vérifier le résultat : taille du `.md` plausible, dossier `*_images/` créé.

### Phase 2 — Structure et découpage

1. Extraire la table des matières : `pymupdf` → `doc.get_toc()` (et `page_count`).
2. Repérer dans le `.md` les **rampeurs** réels des sections (les titres OCR en
   `#`/`##`) avec `grep -n "^#"`, en croisant avec la ToC. Attention : les
   numéros de page résiduels et les en-têtes répétés polluent le grep.
3. Découper par bornes de **lignes** (`sed -n 'A,Bp' f.md > chunks/xxx.md`)
   selon la structure du document : `front.md` (couverture, résumés, ToC),
   `intro.md`, `ch01.md … chNN.md` (un par chapitre), `conclusion.md`,
   `annexe.md` (si annexes de contenu), `tail.md` (bibliographie).
4. **Vérifier la partition exactement** :
   ```bash
   cat chunks/front.md chunks/intro.md ... | diff - manuscrit.md && echo OK
   ```
   Les plages doivent être contiguës et couvrir tout le fichier. Cas piège
   observé : la dernière ligne sans saut de ligne finale (numéro de page
   résiduel) échappe à `sed` → la recoller explicitement et re-vérifier par
   `diff`. Ne pas passer à la phase suivante avant `OK`.

### Phase 3 — Boucle de compression (le cœur)

Pour chaque bout **dans l'ordre**, un à un — ne jamais passer au suivant avant
d'avoir terminé le courant :

1. **Lire intégralement le bout** avec `read_file` en boucle :
   `offset`/`limit` (~300–400 lignes par appel) jusqu'à
   `was_truncated == false`. Ne jamais résumer un bout non lu en entier.
2. **Ajouter** (`>>` heredoc via bash, jamais d'écrasement) une section
   `# ChN — Titre` à `resume.md`.
3. Mettre à jour le fichier d'état du scratchpad + la liste `todo`.

### Règles de compression

- En tête de `resume.md` : une section « Résumé général » issue des résumés
  FR/EN du frontispice (garder UNE langue), + métadonnées (auteurs, jury, etc.)
  + structure du document avec pages.
- Une section par chapitre en sous-sections par §, avec : équations cibles,
  définitions, formules clés en LaTeX (`$…$`), propositions/théorèmes avec leurs
  **hypothèses** et l'**idée de la preuve** en une ligne, références
  croisées (`(2.10)`, `Prop 2.4.6`) pour garder les liens avec la source.
- Résultats numériques : pour chaque test — équation, solution exacte, schémas
  et paramètres, ordres observés (loi de dépendance), conclusions comparatives.
  Ne pas recopier les tableaux bruts.
- Les preuves détaillées : garder le squelette (identifie la technique), pas le
  décalque.
- Fusionner les répétitions (chapitre « Résumé »/« Conclusion » qui recopie :
  écrire « récapitulatif identique aux sections déjà couvertes » + seulement les
  contributions nouvelles).
- Style télégraphique : phrases courtes, « ; » et « → » pour la logique,
  sigles après définition.

### Phase 4 — Vérification finale

1. `grep -n "^# " resume.md` : toutes les sections présentes, dans l'ordre du
   document.
2. Confirmer que chaque bout a été lu (état scratchpad / todo à jour).
3. Signaler à l'utilisateur : position des fichiers (`chunks/`, OCR,
   `resume.md`), taille du résumé, et proposer le nettoyage éventuel des
   intermédiaires (sans rien supprimer sans accord).

## Pièges connus (retour d'expérience)

- `nohup cmd &` dans `bash` ne survit pas au timeout du sandbox → toujours
  `tools.process.start`. On est réveillé par une notification terminal du
  runtime quand le processus se termine.
- Budget d'un programme `run_typescript` ≈ 60 s : pas de longues boucles de
  polling dedans.
- Fin de fichier sans `\n` final : la partition `sed` perd des octets —
  re-vérifier par `diff` octet/octet après réparation.
- Les `read_file` sur un gros bout renvoient `was_truncated: true` :
  continue à `next_offset` jusqu'au bout ; ne jamais interrompre le processus.
- Si le PDF est déjà accompagné d'un `.md` (conversion précédente), la phase 1
  est sautée par le script OCR — exploiter ce fait pour la reprise.
