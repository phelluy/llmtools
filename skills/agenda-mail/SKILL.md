---
name: agenda-mail
description: Ajouter, modifier ou consulter des rendez-vous dans les agendas Thunderbird (CalDAV en mode hors ligne, cache cache.sqlite), avec synchronisation automatique au retour en ligne. À charger quand l'utilisateur demande d'ajouter un rendez-vous, de lire ou de modifier son calendrier Thunderbird.
---

# Agendas Thunderbird

## Où sont les agendas

Les agendas sont déclarés dans `<profil>/prefs.js` (profils snap et classiques :
`~/snap/thunderbird/common/.thunderbird/*`, `~/.thunderbird/*`), clés
`calendar.registry.<id>.{name,type,uri,username,readOnly,cache.enabled}` :

| type | stockage |
|---|---|
| `ics` | fichier ICS local ou distant, souvent en lecture seule |
| `caldav` | serveur distant ; si `cache.enabled=true`, un cache SQLite local |

L'agenda affiché « Calendar » de `partage.unistra.fr` est un `caldav` avec
cache : c'est le cas où l'on peut travailler **hors ligne**.

## Cache local

`<profil>/calendar-data/cache.sqlite`, tables `cal_events` (une ligne par
événement, colonne `cal_id` = id du registre), `cal_properties`
(`item_id`/`key`/`value`/`cal_id` : DESCRIPTION, LOCATION, URL, CLASS,
SEQUENCE, TRANSP), `cal_alarms`, `cal_attendees`, `cal_recurrence`.

Repères de format :

- temps en **microsecondes** (secondes × 10⁶) ;
- colonnes `event_start_tz` / `event_end_tz` : fuseau IANA (`Europe/Paris`) ;
- `flags` empiriques : événement simple = `4` ; `+2` alarme, `+4`
  participants, `+8` pièces jointes, `+16` récurrence ;
- `offline_journal` : `1` = créé hors ligne, `2` = modifié, `3` = supprimé.

## Ajouter un événement en mode hors ligne

Thunderbird **fermé** : le script le vérifie lui-même (`pgrep -x
thunderbird-bin`, doute = refus) et exige `--tb-stopped` — le tromper est
inutile, il refuse s'il trouve le processus. Il découvre l'agenda par son
nom, refuse les agendas `readOnly` ou sans cache, **sauvegarde le fichier**
(`cache.sqlite.bak-<horodatage>`, nom unique), puis insère avec
`offline_journal=1` :

```bash
python3 scripts/insert_offline_event.py --calendar "Calendar" \
    --title "Réunion élus - Préparation CA du 3/11/2026" \
    --start "2026-11-02 18:00" --end "2026-11-02 19:00" \
    --description "..." --location "Visio BBB" \
    --url "https://bbb.unistra.fr/rooms/xxxx/join" --tb-stopped
```

`--end` par défaut = `--start` + 1 h. `--dry-run` montre sans écrire, `--db
/copie.sqlite` permet de tester sur une copie.

## Sauvegardes et retour arrière

```bash
python3 scripts/insert_offline_event.py --calendar "Calendar" --list-backups
python3 scripts/insert_offline_event.py --calendar "Calendar" --tb-stopped --restore
```

`--restore` remet en place la **plus récente** sauvegarde (checkpoint du WAL,
suppression des `-wal`/`-shm`, copie), après vérification que Thunderbird est
fermé et que la sauvegarde est un cache valide. Une fois la synchronisation
confirmée côté serveur ou autre client, supprimer les sauvegardes.

## Sûretés

- Toute écriture exige Thunderbird fermé : `--tb-stopped` + vérification
  automatique par le script (refus si `thunderbird-bin` tourne).
- Sauvegarde datée avant écriture (le script le fait) ; retour arrière par
  `--restore` ; vérifier la synchro côté serveur avant de supprimer les
  sauvegardes.
- Ne jamais modifier un agenda `readOnly`, ni un agenda sans cache actif :
  l'insertion serait écrasée à la resynchronisation.
- Si l'heure de fin n'est pas donnée par l'utilisateur, poser 1 h et le dire.
- Pour lire les agendas sans rien écrire : ouvrir en
  `file:...?mode=ro`, jamais en écriture.
