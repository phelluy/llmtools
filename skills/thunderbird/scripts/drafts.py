#!/usr/bin/env python3
"""Construit les courriels de relecture (brouillons) a partir d'un dossier de reponses.

  python3 drafts.py --dir reponses --attach corrige.pdf
  python3 drafts.py --dir reponses --attach corrige.pdf --install "Brouillons exo11"

Chaque fichier de reponses = un message, avec des en-tetes puis le corps :

    To: DUPOND Marie <marie.dupond@etu.unistra.fr>
    Subject: Re: Rendu exercice 3
    In-Reply-To: <abc123@etu.unistra.fr>

    Bonjour,

    Votre demonstration de la question 2 est complete ; il manque seulement
    le cas nul.
     > python3 exo.py --check
    Bien a vous.

Le corps est rendu en text/plain; format=flowed : les paragraphes sont des lignes
logiques, coupees par des retours *mous* (espace finale) que le client recolle a
sa propre largeur. Les lignes commencant par une espace, par '>' ou situees dans
un bloc ``` restent telles quelles (code, commandes, citations).

Ne jamais utiliser de retour dur a 72 caracteres : en police proportionnelle cela
donne un paragraphe decoupe. C'est le role de format=flowed.

Rien n'est envoye. --install ecrit dans un dossier local Thunderbird (format
mbox), Thunderbird devant etre ferme ; il refuse d'ecraser un fichier existant
et n'ecrit jamais dans l'Outbox ("Messages en attente"), qui serait envoye.
"""
import argparse, base64, glob, hashlib, os, re, subprocess, sys, time, uuid
import quopri
from email.header import Header
from email.utils import formatdate, parsedate_to_datetime

WIDTH = 74


def blocks(text):
    """Corps source -> [(type, contenu)] : 'flux' = paragraphe a recoller,
    'lit' = ligne garde telle quelle (bloc ``` , indentation, citation, vide)."""
    out, para, verbatim = [], [], False
    for raw in text.replace('\r\n', '\n').split('\n'):
        line = raw.rstrip()
        if line.startswith('```'):
            if para:
                out.append(('flux', ' '.join(para))); para = []
            verbatim = not verbatim
            out.append(('lit', line))
            continue
        if verbatim or line[:1] in (' ', '\t', '>'):
            if para:
                out.append(('flux', ' '.join(para))); para = []
            out.append(('lit', raw.rstrip('\r')))
            continue
        if not line.strip():
            if para:
                out.append(('flux', ' '.join(para))); para = []
            out.append(('lit', ''))
            continue
        para.extend(line.split())
    if para:
        out.append(('flux', ' '.join(para)))
    return out


def logical(text):
    return '\n'.join(b if t == 'lit' else b for t, b in blocks(text))


def flowed(text, width=WIDTH):
    """Version format=flowed : les paragraphes sont coupes par des retours mous
    (espace finale) que le client recolle ; les lignes 'lites' ne bougent pas."""
    out = []
    for kind, content in blocks(text):
        if kind == 'lit':
            out.append(content)
            continue
        lines, cur = [], ''
        for word in content.split(' '):
            if cur and len(cur) + 1 + len(word) > width:
                lines.append(cur); cur = word
            else:
                cur = word if not cur else cur + ' ' + word
        lines.append(cur)
        out += [l + ' ' for l in lines[:-1]] + [lines[-1]]
    return '\n'.join(out)


def unflow(text):
    """Recolle les retours mous : sert a verifier qu'aucun mot ne manque."""
    out, buf = [], ''
    for line in text.split('\n'):
        soft = buf.endswith(' ')
        if buf and not soft:
            out.append(buf); buf = ''
        if line.startswith((' ', '\t', '>')) or line == '':
            if buf:
                out.append(buf.rstrip(' ')); buf = ''
            out.append(line)
            continue
        buf = (buf.rstrip(' ') + ' ' + line) if buf else line
    if buf:
        out.append(buf.rstrip(' '))
    return '\n'.join(out)


def parse_file(path):
    txt = open(path, encoding='utf-8').read().replace('\r\n', '\n')
    head = {}
    i = 0
    lines = txt.split('\n')
    while i < len(lines) and lines[i].strip() and ':' in lines[i] and not lines[i].startswith(' '):
        k, v = lines[i].split(':', 1)
        if k.strip().lower() not in ('to', 'cc', 'subject', 'in-reply-to', 'references', 'from'):
            break
        head[k.strip().lower()] = v.strip()
        i += 1
    return head, '\n'.join(lines[i:]).strip('\n')


def build(head, body, attaches, from_addr, boundary):
    mid = '<%s@unistra.fr>' % uuid.uuid4()
    hdr = [('From', from_addr), ('To', head['to']), ('Date', formatdate(localtime=True)),
           ('Message-ID', mid)]
    if head.get('subject'):
        hdr.append(('Subject', Header(head['subject'], 'utf-8').encode()))
    if head.get('in-reply-to'):
        refs = head.get('references') or head['in-reply-to']
        hdr += [('In-Reply-To', head['in-reply-to']), ('References', refs)]
    hdr += [('User-Agent', 'Thunderbird'), ('MIME-Version', '1.0'),
            ('Content-Type', 'multipart/mixed; boundary="%s"' % boundary)]
    b = b'\r\n'.join(('%s: %s' % (k, v)).encode('utf-8') for k, v in hdr) + b'\r\n\r\n'
    fl = flowed(body)
    b += ('--%s\r\nContent-Type: text/plain; charset=UTF-8; format=flowed\r\n'
          'Content-Transfer-Encoding: quoted-printable\r\n\r\n' % boundary).encode()
    b += b'\r\n'.join(quopri.encodestring(l.encode('utf-8')).rstrip(b'\r\n')
                      for l in fl.split('\n')) + b'\r\n'
    for path in attaches:
        data = open(path, 'rb').read()
        b += ('--%s\r\nContent-Type: application/octet-stream; name="%s"\r\n'
              'Content-Transfer-Encoding: base64\r\n'
              'Content-Disposition: attachment; filename="%s"\r\n\r\n'
              % (boundary, os.path.basename(path), os.path.basename(path))).encode()
        b += base64.encodebytes(data) + b'\r\n'
    b += ('--%s--\r\n' % boundary).encode()
    return b, fl


def tb_running():
    """Faux seulement si Thunderbird est vraiment absent. On teste le *nom* de
    processus exactement (pgrep -x) : pgrep -f lit les lignes de commande et
    finirait par matcher le shell qui nous appelle. En cas de doute : True."""
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


def local_dir():
    for pat in ('~/snap/thunderbird/common/.thunderbird/*/Mail/Local Folders',
                '~/.thunderbird/*/Mail/Local Folders',
                '~/snap/thunderbird/*/common/.thunderbird/*/Mail/Local Folders',
                '~/Library/Thunderbird/Profiles/*/Mail/Local Folders'):
        for d in glob.glob(os.path.expanduser(pat)):
            if os.path.isdir(d):
                return d
    raise SystemExit('dossier "Local Folders" introuvable')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='reponses')
    ap.add_argument('--attach', action='append', default=[])
    ap.add_argument('--from', dest='fromaddr', default='philippe.helluy@unistra.fr')
    ap.add_argument('--out', default='brouillons')
    ap.add_argument('--install', metavar='NOM_DU_DOSSIER', default=None)
    ap.add_argument('--tb-stopped', action='store_true',
                    help='affirme que Thunderbird est ferme (requis avec --install)')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, '*.txt')) + glob.glob(os.path.join(a.dir, '*.md')))
    if not files:
        raise SystemExit('aucun fichier de reponse dans %s/' % a.dir)
    for p in a.attach:
        if not os.path.exists(p):
            raise SystemExit('piece jointe absente : %s' % p)
    os.makedirs(a.out, exist_ok=True)
    boundary = '----=_Part_' + uuid.uuid4().hex
    chunks, report = [], []
    for i, f in enumerate(files, 1):
        head, body = parse_file(f)
        if 'to' not in head:
            raise SystemExit('%s : en-tete To: manquant' % f)
        raw, fl = build(head, body, a.attach, a.fromaddr, boundary)
        if unflow(fl) != logical(body):
            raise SystemExit('aller-retour flowed perdu pour %s : ne pas livrer' % f)
        base = os.path.splitext(os.path.basename(f))[0]
        eml = os.path.join(a.out, '%02d_%s.eml' % (i, re.sub(r'^\d+_|\.md$', '', base)))
        open(eml, 'wb').write(raw)
        chunks.append(raw)
        report.append((os.path.basename(eml), head['to'], head.get('subject', ''), len(body)))
    stamp = (time.strftime('%a %b ') + ('%2d' % time.localtime().tm_mday)
             + time.strftime(' %H:%M:%S %Y')).encode()
    blob = b''.join(b'From - ' + stamp + b'\r\n'
                    b'X-Mozilla-Status: 0003\nX-Mozilla-Status2: 00000000\n'
                    + re.sub(b'(?m)^From ', b'>From ', c) + b'\r\n' for c in chunks)
    mbox = os.path.join(a.out, 'dossier_local.mbox')
    open(mbox, 'wb').write(blob)
    for r in report:
        print('%-34s %-46s %5d car  %s' % (r[0], r[1], r[3], r[2][:40]))
    print('%d message(s) -> %s/ + %s' % (len(report), a.out, mbox))

    if a.install:
        if not a.tb_stopped:
            raise SystemExit('--install exige --tb-stopped : faites fermer Thunderbird '
                             'par l\'utilisateur, verifiez, puis relancez avec ce drapeau.')
        if tb_running():
            raise SystemExit("Thunderbird est en cours d'execution : fermez-le avant --install.")
        target = os.path.join(local_dir(), a.install)
        if os.path.exists(target):
            raise SystemExit('%s existe deja : choisissez un autre nom (on n'
                             "'ecrase pas un dossier existant)" % target)
        open(target, 'wb').write(blob)
        os.chmod(target, 0o600)
        print('installe dans', target, '-', len(blob), 'octets')


if __name__ == '__main__':
    main()
