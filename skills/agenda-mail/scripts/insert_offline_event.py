#!/usr/bin/env python3
"""Ajoute un evenement dans le cache d'un agenda Thunderbird, en mode hors ligne.

Thunderbird ferme. L'evenement est insere dans cache.sqlite avec
offline_journal=1 (cree hors ligne) : au prochain passage en ligne, la
synchronisation CalDAV le pousse au serveur. Une sauvegarde du fichier est
faite avant toute ecriture.

  python3 insert_offline_event.py --calendar "Calendar" \
      --title "Reunion CA" --start "2026-11-02 18:00" --end "2026-11-02 19:00" \
      --description "..." --location "Visio BBB" \
      --url "https://bbb.unistra.fr/rooms/xxxx/join"

Ne jamais ecrire si Thunderbird tourne : --tb-stopped atteste qu'il est
ferme et le script le verifie lui-meme (pgrep -x thunderbird-bin). Avant toute
ecriture, une sauvegarde datee cache.sqlite.bak-<TS> est faite ; --restore la
remet en place.

  python3 insert_offline_event.py --calendar "Calendar" \
      --title "Reunion CA" --start "2026-11-02 18:00" --end "2026-11-02 19:00" \
      --description "..." --location "Visio BBB" \
      --url "https://bbb.unistra.fr/rooms/xxxx/join" --tb-stopped

  python3 insert_offline_event.py --calendar "Calendar" --tb-stopped --restore
  python3 insert_offline_event.py --calendar "Calendar" --list-backups

  --db /tmp/test.sqlite   : travailler sur une copie (tests)
  --dry-run               : montrer les lignes sans ecrire
"""
import argparse, glob, os, re, shutil, sqlite3, subprocess, sys, time, uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def tb_running():
    """Faux seulement si Thunderbird est vraiment absent. On teste le *nom* de
    processus exactement (pgrep -x) : pgrep -f lirait les lignes de commande et
    matcherait le shell appelant. En cas de doute : True."""
    try:
        r = subprocess.run(['pgrep', '-x', 'thunderbird-bin'],
                           capture_output=True, text=True)
        if r.stdout.strip():
            return True
        if r.returncode == 1:
            return False
    except Exception:
        pass
    for path in glob.glob('/proc/[0-9]*/cmdline'):
        try:
            if int(path.split('/')[2]) == os.getpid():
                continue
            c = open(path, 'rb').read()
            if b'thunderbird' in c.lower() and b'sh' not in os.path.basename(c.split(b'\x00')[0].decode() or 'x'):
                return True
        except Exception:
            continue
    return True


def profiles():
    out = []
    for pat in ('~/snap/thunderbird/common/.thunderbird/*', '~/.thunderbird/*',
                '~/snap/thunderbird/*/common/.thunderbird/*'):
        for p in glob.glob(os.path.expanduser(pat)):
            if os.path.exists(os.path.join(p, 'prefs.js')):
                out.append(p)
    return out


def registry(prefs_path):
    """calendar.registry.<id>.<cle> -> {id: {cle: valeur}}"""
    data = open(prefs_path, encoding='utf-8', errors='replace').read()
    cals = {}
    rx = re.compile(r'user_pref\("calendar\.registry\.([0-9a-fA-F-]{36})\.([\w.]+)",'
                    r'\s*(?:"((?:[^"\\]|\\.)*)"|(true|false|[0-9]+|[A-Za-z-]+))\)')
    for cid, key, quoted, bare in rx.findall(data):
        val = quoted.replace('\\"', '"') if quoted else bare
        cals.setdefault(cid.lower(), {})[key] = val
    return cals


def find_calendar(name, profile=None):
    profs = [profile] if profile else profiles()
    for p in profs:
        for cid, meta in registry(os.path.join(p, 'prefs.js')).items():
            if meta.get('name', '').strip().lower() == name.strip().lower():
                return p, cid, meta
    return None, None, None


def parse_local(s, tz):
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        raise SystemExit(f"date illisible : {s!r} (attendu 'YYYY-MM-DD HH:MM')")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.astimezone(ZoneInfo('UTC'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calendar', required=True, help='nom affiche de l\'agenda')
    ap.add_argument('--title', default='', help='titre de l\'evenement')
    ap.add_argument('--start', default='', help="'YYYY-MM-DD HH:MM' (heure locale)")
    ap.add_argument('--end', default='', help="defaut : start + 1 h")
    ap.add_argument('--description', default='')
    ap.add_argument('--location', default='')
    ap.add_argument('--url', default='')
    ap.add_argument('--tz', default='Europe/Paris')
    ap.add_argument('--profile', default='', help='profil Thunderbird explicite')
    ap.add_argument('--db', default='', help='remplace le chemin de cache.sqlite (tests)')
    ap.add_argument('--tb-stopped', action='store_true',
                    help='atteste que Thunderbird est ferme (requis pour ecrire)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--list-backups', action='store_true',
                    help='lister les sauvegardes disponibles')
    ap.add_argument('--restore', action='store_true',
                    help='remettre la derniere sauvegarde en place')
    a = ap.parse_args()

    prof, cid, meta = find_calendar(a.calendar, a.profile or None)
    if not cid:
        raise SystemExit(f"agenda {a.calendar!r} introuvable dans prefs.js")
    dbpath = a.db or os.path.join(prof, 'calendar-data', 'cache.sqlite')

    def check_tb():
        if not a.tb_stopped:
            raise SystemExit("ecriture refusee : passez --tb-stopped une fois Thunderbird ferme")
        if tb_running():
            raise SystemExit("Thunderbird est en cours d'execution : fermez-le avant d'ecrire.")

    backups = sorted(glob.glob(dbpath + '.bak-*'))
    if a.list_backups:
        print(f"{len(backups)} sauvegarde(s) pour {dbpath} :")
        for b in backups:
            print(' ', b, os.path.getsize(b), 'octets')
        return
    if a.restore:
        if not backups:
            raise SystemExit('aucune sauvegarde *.bak-* a restaurer')
        check_tb()
        bak = backups[-1]
        chk = sqlite3.connect(f'file:{bak}?mode=ro', uri=True)
        ok = chk.execute("select count(*) from sqlite_master where name='cal_events'").fetchone()[0]
        chk.close()
        if not ok:
            raise SystemExit(f"sauvegarde illisible ou invalide : {bak}")
        db = sqlite3.connect(dbpath)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        db.close()
        for side in ('-wal', '-shm'):
            if os.path.exists(dbpath + side):
                os.remove(dbpath + side)
        shutil.copy2(bak, dbpath)
        print(f'restaure : {bak} -> {dbpath}')
        print('relancez Thunderbird et verifiez l\'agenda.')
        return

    if not a.title or not a.start:
        raise SystemExit("--title et --start sont requis pour un ajout")
    print(f"agenda : {meta.get('name')} [{meta.get('type')}] cache={meta.get('cache.enabled')} "
          f"readOnly={meta.get('readOnly')} cal_id={cid}")
    if meta.get('readOnly', 'false') == 'true':
        raise SystemExit('agenda en lecture seule : refuse')
    if meta.get('type') != 'caldav' or meta.get('cache.enabled') != 'true':
        raise SystemExit("seuls les agendas caldav avec cache actif sont modifies "
                         "en mode hors ligne (sinon l'insertion sera ecrasee a la sync)")
    if not os.path.exists(dbpath):
        raise SystemExit(f"pas de cache : {dbpath}")

    start = parse_local(a.start, a.tz)
    end = parse_local(a.end, a.tz) if a.end else start + timedelta(hours=1)
    if end <= start:
        raise SystemExit('fin avant le debut')

    ro = sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True)
    n = ro.execute("select count(*) from cal_events where cal_id=?", (cid,)).fetchone()[0]
    ro.close()
    if not n:
        raise SystemExit(f"cal_id {cid} absent du cache : mauvais agenda ou cache vide")

    now = int(time.time() * 1_000_000)
    eid = str(uuid.uuid4()).upper()
    us = lambda dt: int(dt.timestamp() * 1_000_000)
    props = {'CLASS': 'PUBLIC', 'SEQUENCE': '0', 'TRANSP': 'OPAQUE'}
    if a.description:
        props['DESCRIPTION'] = a.description
    if a.location:
        props['LOCATION'] = a.location
    if a.url:
        props['URL'] = a.url

    print(f"evenement : {a.title}")
    print(f"  {start.astimezone(ZoneInfo(a.tz)):%A %d %B %Y %H:%M} -> "
          f"{end.astimezone(ZoneInfo(a.tz)):%H:%M} ({a.tz}), id {eid}")
    if a.dry_run:
        print('dry-run : rien ecrit')
        return

    check_tb()
    bak = f"{dbpath}.bak-{time.strftime('%Y%m%d%H%M%S')}"
    n = 0
    while os.path.exists(bak):
        n += 1
        bak = f"{dbpath}.bak-{time.strftime('%Y%m%d%H%M%S')}-{n}"
    shutil.copy2(dbpath, bak)
    print(f"sauvegarde : {bak}")

    db = sqlite3.connect(dbpath)
    try:
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        db.execute('begin')
        db.execute("""insert into cal_events
            (cal_id, id, time_created, last_modified, title, priority, privacy,
             ical_status, flags, event_start, event_end, event_stamp,
             event_start_tz, event_end_tz, recurrence_id, recurrence_id_tz,
             alarm_last_ack, offline_journal)
            values (?,?,?,?,?,?, 'PUBLIC','CONFIRMED', 4,?,?,?,?,?,
                    NULL,NULL,NULL, 1)""",
            (cid, eid, now, now, a.title, None,
             us(start), us(end), now, a.tz, a.tz))
        for key, val in props.items():
            db.execute("""insert into cal_properties
                (item_id, key, value, recurrence_id, recurrence_id_tz, cal_id)
                values (?,?,?,NULL,NULL,?)""", (eid, key, val, cid))
        db.commit()
    except Exception:
        db.rollback()
        db.close()
        raise
    chk = db.execute("select title, offline_journal from cal_events where id=?", (eid,)).fetchone()
    db.close()
    print(f"insere : {chk[0]!r} offline_journal={chk[1]}")
    print('Au retour en ligne, Thunderbird pousse l\'evenement au serveur ;')
    print('verifie sur un autre client, puis supprime la sauvegarde.')


if __name__ == '__main__':
    main()
