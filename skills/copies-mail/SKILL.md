---
name: copies-mail
description: Corriger des travaux d'étudiants rendus par courriel (maths, informatique, tout niveau) - retrouver les rendus dans Thunderbird, fusionner en un fichier par étudiant, vérifier sans écrire, produire un corrigé bref sans note, rédiger les réponses en brouillon. À charger quand l'utilisateur demande de corriger des copies envoyées par mail.
---

# Corriger des copies rendues par courriel

La mécanique Thunderbird (lire les mbox, extraire les pièces, poser les
brouillons) est dans le skill **`thunderbird`** — le charger si besoin. Ses
scripts vivent dans `~/.vibe/skills/thunderbird/scripts/` : `collect.py`
(retrouver et extraire) et `drafts.py` (construire et installer les brouillons).
Ici ne restent que la fusion des copies et les règles métier.

## Règles de fond (non négociables)

- **Aucune note, aucun barème, aucun point** dans les courriels ni dans le
  corrigé. On explique ce qui tient, ce qui manque, et quoi pousser.
- Ton **bienveillant et factuel** : une louange *précise* (ce que la copie fait
  bien, en citant la ligne/le raisonnement), puis **une seule** piste à pousser.
- Le corps des messages est en `format=flowed` : paragraphes sur une ligne
  logique, coupés par des retours *mous*. **Jamais** de retour dur à 72
  caractères.
- Les adresses viennent du **courriel d'origine**, jamais d'une liste : on
  répond à l'étudiant qui a envoyé, avec son sujet en « Re: » et l'`In-Reply-To`
  d'origine.
- Rien n'est jamais envoyé.

## Déroulé

### 0. Cadre

Travailler **dans le dossier de l'exercice** (un dossier par sujet). Récupérer
le **lien web du sujet** : il donne l'énoncé exact et borne la période d'envoi.
Demander le lien et la fenêtre d'envoi s'ils manquent ; noter les deux dans
`sujet.md` au dossier.

### 1. Retrouver les courriels

```bash
python3 ~/.vibe/skills/thunderbird/scripts/collect.py \
        --since "2026-10-01 00:00" --until "2026-10-08 08:00" \
        --from-regex '@etu\.' --list-only
```

Puis `--extract` quand la fenêtre est bonne. Vérifier la couverture : comparer
le nombre de rendus avec la liste d'émargement du TD, chercher explicitement
les absents avant d'écrire « je n'ai pas reçu votre copie ». Un envoi hors
fenêtre (nuit du rendu) est courant : élargir d'une dizaine d'heures avant de
conclure.

### 2. Une copie = un fichier

```bash
python3 scripts/merge.py
```

`NOM_Prenom__*` → **un seul fichier** par étudiant : `NOM_Prenom.md` si tout
est texte, `NOM_Prenom.pdf` s'il y a une image ou un PDF (composé avec
`typst`). Les originaux restent en place. Un conteneur est d'abord dégradé :
`.zip` vidé, `.sqbpro` lu comme XML (onglets `<sql name=...>`), `.ipynb` déplié
en cellules. Si un fichier est illisible (format propriétaire, image scannée),
le **dire** dans le message plutôt que de deviner son contenu.

### 3. Vérifier avant d'écrire

Exécuter / recalculer, **jamais en écriture** : base SQLite en
`file:...?mode=ro`, code d'étudiant dans un répertoire temporaire, commandes
sans effet de bord. Reproduire soi-même le résultat attendu avant d'affirmer
qu'une réponse est juste ou fausse. Pour une question difficile, écrire le
programme de référence plutôt que de raisonner de tête. Distinguer ce qui est
**demandé**, ce qui est **calculé**, ce qui est **vrai**.

### 4. Le corrigé

Un PDF de **une à deux pages**, sans note : énoncé en une phrase, réponse à
chaque question, le point délicat (celui qui sépare les copies), et la variante
qui change le résultat si l'on lâche une hypothèse. Généré avec `typst` (skill
`typst`), nommé `corrige_<sujet>.pdf`. Le mentionner dans chaque message.

### 5. Les courriels

Un fichier par étudiant dans `reponses/NN_NOM_Prenom.md` :

```
To: DUPOND Marie <marie.dupond@etu.unistra.fr>
Subject: Re: Rendu exercice 3
In-Reply-To: <1562248108.32693656@etu.unistra.fr>

Bonjour,

merci pour votre rendu. <ce qui tient, en citant précisément>

<le point de repère : la bonne réponse, et pourquoi>

<une seule piste à pousser>

Le corrigé est en pièce jointe. Si quelque chose reste obscur, passez me voir.

ph
```

Une ligne = un paragraphe (pas de retour dur). Une ligne commençant par une
espace, par `>`, ou à l'intérieur d'un bloc ``` reste telle quelle.

### 6. Poser les brouillons

```bash
python3 ~/.vibe/skills/thunderbird/scripts/drafts.py --dir reponses --attach corrige.pdf \
        --install "Brouillons exo11" --tb-stopped
```

Thunderbird **fermé**. Puis l'utilisateur sélectionne tout et recopie vers le
dossier **Brouillons** du compte (c'est ce copier-coller qui fait l'APPEND).

## Contrôles avant de conclure

- Une copie = un fichier fusionné, et un message par étudiant ayant envoyé.
- Chaque message : bon destinataire, sujet en « Re: », `In-Reply-To`, pièce
  jointe identique octet pour octet au corrigé.
- Aucun mot « note », « points », « barème », « sur 20 ».
- Rien n'a été envoyé ; aucun fichier existant du profil n'a bougé.
