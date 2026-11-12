# Thunderbird : stores locaux, sûretés, plan B

## Trouver les dossiers

Le profil peut être confiné en snap :

```
~/snap/thunderbird/common/.thunderbird/<hash>.default/     # snap (Linux)
~/.thunderbird/<hash>.default/                             # classique (Linux)
~/Library/Thunderbird/Profiles/<hash>.default/             # macOS
```

Sur macOS il n'y a pas de snap : la racine est `~/Library/Thunderbird/`
(`profiles.ini` et sous-dossier `Profiles/` contenant les profils). Une fois
le profil trouvé, l'organisation interne est **identique** aux trois endroits
(`prefs.js`, `Mail/`, `ImapMail/`), donc toute la suite s'applique telle quelle.

`prefs.js` donne tout :

| Clé | Sens |
|---|---|
| `mail.server.serverN.directory` | dossier qui contient les mbox d'un compte |
| `mail.server.serverN.type` | `imap`, ou `none` pour « Dossiers locaux » |
| `mail.accountmanager.localfoldersserver` | le serveur des dossiers locaux |
| `mail.identity.idN.draft_folder` | le dossier Brouillons utilisé par une identité |
| `mail.identity.idN.useremail` | l'adresse à mettre dans `From:` |

Un compte IMAP a un dossier = un fichier sans extension dans
`ImapMail/<serveur>/`, avec son `.msf` à côté. Le nom de fichier peut différer
du nom affiché (`Drafts` peut être `Drafts-1` après un collision). Un nom non
ASCII est translittéré (`Réseaux sociaux` → `R&AOk-seaux sociaux`) : ne pas
deviner, lister le répertoire.

Les dossiers locaux sont dans `Mail/Local Folders/`, dont `Unsent Messages` =
l'Outbox (« Messages en attente ») : **à ne jamais alimenter**.

## Deux formats de message local

Dossiers **locaux** (file-based) : `From - Thu Oct  8 11:13:24 2026` (jour
espacé, pas de virgule), puis les en-têtes Mozilla finissant par `\n` (LF seul),
puis le message en CRLF. Thunderbird reconstruit l'index depuis le mbox quand
le `.msf` manque ou que la taille a changé — c'est pour ça qu'un dossier local
**ajouté** à la main est fiable.

Dossiers **IMAP** : même aspect, mais l'index `.msf` fait autorité et la boîte
est reconciliée avec le serveur à la reconnexion. Un message ajouté à la main
peut être ignoré puis supprimé silencieusement. **Ne pas écrire dans un cache
IMAP.**

`X-Mozilla-Status` : `0001` nouveau/lu, `+0002` = brouillon, `+0010` et
`+0020` = réponses/transferts. `0003` = nouveau **et** brouillon : double-clic
ouvre la fenêtre de rédaction, pièce jointe comprise ; `0001` se lit dans le
volet de lecture. Pour une relecture suivie d'un envoi, `0003`.

## Sûretés

1. Thunderbird **fermé** avant toute écriture. Le demander à l'utilisateur, puis
   vérifier qu'aucun `thunderbird-bin` ne tourne. En cas de doute, considérer
   qu'il tourne. Note : tester le *nom* de processus exactement (`pgrep -x thunderbird-bin`),
jamais `pgrep -f thunderbird` : la ligne de commande du shell appelant
contient le mot cherche et pgrep la matcherait. D'où aussi le drapeau
explicite `--tb-stopped`.
2. **Ajouter** un fichier, n'en modifier aucun. Vérifier après coup que les
   mbox et `.msf` existants ont la même taille et la même date qu'avant.
3. Un nouveau dossier local porte un nom **ASCII sans accent** (sinon Thunderbird
   le rename de façon opaque).
4. Rien n'est envoyé, jamais. La sortie attendue d'un livrage est : brouillons
   visibles, doigt sur la souris de l'utilisateur.

## Plan B quand le dossier n'apparaît pas

Les `.eml` produits sont des messages complets autonomes. Dans Thunderbird :
sélectionner les fichiers dans le gestionnaire de fichiers et les **glisser** dans
le dossier visé (l'opération fait l'APPEND), ou les ouvrir puis
« Transférer vers ». C'est aussi la voie la plus courte vers le dossier
Brouillons d'un compte distant : copier-coller depuis le dossier local.

## Vérifier un livrage

```bash
ls -la "<profil>/Mail/Local Folders/"                 # nouveau fichier, rien d'autre n'a bougé
grep -ac '^From - ' "<profil>/Mail/Local Folders/XX"  # nb de messages == nb d'etudiants
```

Pour un compte IMAP après le copier-coller de l'utilisateur, le fichier
`Drafts*` local grossit d'environ la taille des messages : c'est la trace que
l'APPEND a eu lieu.
