#!/usr/bin/env python3
r"""Retrouve les rendus d'etudiants dans les boites Thunderbird et les extrait.

Lecture seule stricte : le script ne fait qu'ouvrir les fichiers mbox et ecrire
dans le dossier courant. Il ne deplace, ne modifie et ne supprime aucun message.

  python3 collect.py --since "2026-10-01 00:00" --until "2026-10-08 08:00" \
                     --from-regex '@etu\.' --extract
  python3 collect.py --boxes /chemin/INBOX --since ... --list-only

Sans --boxes, les magasins sont decouverts depuis le profil Thunderbird
(prefs.js -> mail.server.serverN.directory), profils snap et classiques.
Sortie : ./index_rendus.json (+ les pieces jointes prefixees NOM_Prenom__ si --extract)
"""
import argparse, email, email.policy, glob, hashlib, json, os, re, subprocess
from datetime import datetime, timezone
from email.header import decode_header, make_header

TXT = {'.txt', '.md', '.tex', '.sql', '.py', '.c', '.h', '.cpp', '.java', '.rs',
       '.m', '.jl', '.r', '.csv', '.json', '.yaml', '.yml', '.html', '.ipynb',
       '.sqbpro', '.sage', '.m2', '.gp', '.ml', '.pl', '.sh', '.asm', '.f90'}
IMG = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff'}
BIN = {'.pdf', '.zip', '.odt', '.docx', '.7z', '.rar', '.gz', '.tar', '.xlsx', '.doc'}


def profiles():
    """Tous les repertoires de stockage de messages du (des) profil(s)."""
    boxes = []
    cand = []
    for pat in ('~/snap/thunderbird/common/.thunderbird/*', '~/.thunderbird/*',
                '~/snap/thunderbird/*/common/.thunderbird/*',
                '~/Library/Thunderbird/Profiles/*'):
        cand += glob.glob(os.path.expanduser(pat))
    for p in cand:
        prefs = os.path.join(p, 'prefs.js')
        if not os.path.exists(prefs):
            continue
        with open(prefs, 'rb') as f:
            data = f.read()
        for m in re.finditer(rb'"mail\.server\.server\d+\.(?:directory|directory-rel)", "([^"]+)"', data):
            v = m.group(1).decode()
            v = v.replace('[ProfD]', p)
            if '{' in v:                      # multi-compte, on ignore les variants
                continue
            d = v if os.path.isabs(v) else os.path.join(p, v)
            if os.path.isdir(d):
                for root, dirs, files in os.walk(d):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        if fn.endswith(('.msf', '.dat')) or fn.startswith('.'):
                            continue
                        try:
                            if os.path.getsize(fp) > 200:
                                boxes.append(fp)
                        except OSError:
                            pass
    return sorted(set(boxes))


def hdr(msg, name):
    v = msg.get(name)
    if v is None:
        return ''
    try:
        return str(make_header(decode_header(v)))
    except Exception:
        return v


def msgdate(block):
    m = re.search(rb'(?m)^Date: ((?:[^\r\n]|[\r\n][ \t])*)', block)
    if not m:
        m = re.search(rb'^From - [^\r\n]*', block)
        if m:
            try:
                return datetime.strptime(m.group(0)[7:].decode()[:24], '%a %b %d %H:%M:%S %Y'
                                         ).replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                return 0
        return 0
    try:
        return email.utils.parsedate_to_datetime(
            m.group(1).decode('utf-8', 'replace').replace('\r\n', ' ')).timestamp()
    except Exception:
        return 0


def headers(block):
    """En-tetes d'un bloc, sans lire le corps (rapide sur les gros mbox)."""
    head = block[:20000]
    out = {}
    for k in (b'From', b'To', b'Subject', b'Message-ID', b'Date'):
        m = re.search(rb'(?m)^' + k + rb': ((?:[^\r\n]|[\r\n][ \t])*)', head)
        out[k.decode()] = str(make_header(decode_header(
            m.group(1).decode('utf-8', 'replace').replace('\r\n', ' ')))) if m else ''
    return out


def prev_boundary(f, end, chunk=4 << 20):
    """Offset du debut du message qui se termine a `end`, None si indefinissable."""
    pos = end
    while pos > 0:
        n = min(chunk, pos)
        pos -= n
        f.seek(pos)
        blk = f.read(n)
        i = blk.rfind(b'\nFrom - ')
        if i >= 0:
            return pos + i + 1
    return 0 if f.tell() == 0 else None


def scan(path, lo, hi, patience=400, cap=20000):
    """Remonte le mbox message par message depuis la fin, s'arrete apres
    `patience` messages trop vieux. Ne lit que le debut de chaque message."""
    size = os.path.getsize(path)
    found, too_old, end = [], 0, size
    with open(path, 'rb') as f:
        while end > 0:
            start = prev_boundary(f, end)
            if start is None:
                break
            span = end - start
            f.seek(start)
            head = f.read(min(span, 1 << 16))
            end = start
            ts = msgdate(head)
            if hi and ts > hi:
                continue
            if lo and ts < lo:
                too_old += 1
                if too_old >= patience:
                    break
                continue
            too_old = 0
            h = headers(head)
            found.append(dict(box=path, off=start, size=span, ts=ts,
                              date=datetime.fromtimestamp(ts).isoformat() if ts else '',
                              frm=h['From'], to=h['To'], subj=h['Subject'], msgid=h['Message-ID']))
            if len(found) >= cap:
                break
    return found


def student_key(frm):
    """'DUPOND Marie (ETU MAI)' -> 'DUPOND_Marie' ; sinon l'adresse."""
    m = re.match(r'\s*"?([^"<]*?)"?\s*<', frm)
    name = (m.group(1) if m else frm).strip()
    name = re.sub(r'\((?:ETU|UEF|PRAG|ADR)?\s*[A-Z]*\)', '', name).strip(' ,;')
    parts = [p for p in re.split(r'[\s_]+', name) if p]
    if len(parts) >= 2:
        return '_'.join([parts[0].upper()] + [p.capitalize() for p in parts[1:-1]] + [parts[-1]])
    return re.sub(r'[^A-Za-z0-9]+', '_', re.sub(r'@.*', '', name)) or 'inconnu'


def parse_msg(path, off, size):
    with open(path, 'rb') as f:
        f.seek(off)
        raw = f.read(min(size, 60 << 20))
    raw = re.sub(rb'(?m)^>From ', b'From ', raw)
    return email.message_from_bytes(raw, policy=email.policy.default)


def save(data, outdir, key, fname):
    safe = re.sub(r'[\\/:*?"<>|]+', '_', fname).strip('.') or 'piecejointe'
    p = os.path.join(outdir, f'{key}__{safe}')
    n = 1
    while os.path.exists(p):
        p = os.path.join(outdir, f'{key}__{n}_{safe}')
        n += 1
    with open(p, 'wb') as f:
        f.write(data)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since', required=True)
    ap.add_argument('--until', default='')
    ap.add_argument('--boxes', nargs='*', default=None)
    ap.add_argument('--from-regex', default=r'@etu\.|@etu-')
    ap.add_argument('--subject-regex', default='')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--patience', type=int, default=400)
    ap.add_argument('--list-only', action='store_true')
    a = ap.parse_args()

    def ts(s):
        return datetime.strptime(s, '%Y-%m-%d %H:%M').timestamp() if s else 0

    lo, hi = ts(a.since), ts(a.until)
    boxes = a.boxes or profiles()
    if not boxes:
        raise SystemExit('aucun magasin Thunderbird trouve : passez --boxes')
    allm = []
    for b in boxes:
        allm += scan(b, lo, hi, a.patience)
    allm.sort(key=lambda m: m['ts'])
    rx_from = re.compile(a.from_regex, re.I)
    rx_subj = re.compile(a.subject_regex, re.I) if a.subject_regex else None
    keep = [m for m in allm if rx_from.search(m['frm'])
            and (not rx_subj or rx_subj.search(m['subj']))]
    if a.list_only:
        for m in keep:
            print(m['date'], '|', m['frm'], '|', m['subj'][:60], '|', m['msgid'])
        print(len(keep), 'messages retenus /', len(allm), 'scannes')
        return
    index = []
    for m in keep:
        msg = parse_msg(m['box'], m['off'], m['size'])
        key = student_key(m['frm'])
        files = []
        for part in msg.walk():
            if part.is_multipart():
                continue
            fname = part.get_filename()
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            if fname:
                files.append(save(payload, a.outdir, key, fname))
            elif part.get_content_type() == 'text/plain' and len(payload) > 40:
                files.append(save(payload, a.outdir, key, 'corps_message.txt'))
        body = ''
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' and not part.get_filename():
                try:
                    body = part.get_content()
                except Exception:
                    continue
                break
        index.append(dict(student=key, from_=m['frm'], to=m['to'], date=m['date'],
                          subject=m['subj'], message_id=m['msgid'], files=files,
                          corps=body[:1500]))
    with open(os.path.join(a.outdir, 'index_rendus.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(f"{len(index)} rendus -> index_rendus.json")
    for r in index:
        exts = sorted({os.path.splitext(p)[1].lower() for p in r['files']})
        print(f"  {r['student']:<28} {r['date'][:16]}  {len(r['files'])} fichier(s) {' '.join(exts)}")


if __name__ == '__main__':
    main()
