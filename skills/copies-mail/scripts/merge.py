#!/usr/bin/env python3
"""Reunit les fichiers d'un meme etudiant en UNE copie lisible.

  python3 merge.py                      # traite NOM_Prenom__* du dossier courant
  python3 merge.py --files a.py b.png   # ou une liste explicite

Regle : tout est texte  ->  NOM_Prenom.md   (blocs de code, un par fichier)
        il y a une image ou un PDF  ->  NOM_Prenom.pdf  (compose avec typst)

Les fichiers d'origine sont conserves intacts ; seule la copie fusionnee est
ajouree. Un conteneur (.zip, .sqbpro, .ipynb) est d'abord degrade en texte.
"""
import argparse, glob, json, os, re, shutil, subprocess, tempfile, zipfile

TXT = {'.txt', '.md', '.tex', '.sql', '.py', '.c', '.h', '.cpp', '.java', '.rs', '.m',
       '.jl', '.r', '.csv', '.json', '.yaml', '.yml', '.html', '.log', '.sage', '.gp',
       '.ml', '.pl', '.sh', '.f90', '.adb', '.ada', '.go', '.ts', '.js', '.mma', '.nb'}
IMG = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff'}
CODE = {'.py': 'python', '.sql': 'sql', '.c': 'c', '.h': 'c', '.cpp': 'cpp', '.java': 'java',
        '.rs': 'rust', '.m': 'matlab', '.jl': 'julia', '.r': 'r', '.tex': 'latex',
        '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json', '.sh': 'bash', '.go': 'go',
        '.f90': 'fortran', '.pl': 'perl', '.html': 'html'}
NOEXT = {'.sqbpro': 'sql', '.dat': 'text'}


def read_text(path):
    for enc in ('utf-8', 'utf-16', 'latin-1'):
        try:
            return open(path, encoding=enc).read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return None


def degrade(path, workdir):
    """Conteneur -> (type, [(nom, contenu_texte)], [images], [pdf])"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.zip':
        out, d = [], tempfile.mkdtemp(dir=workdir)
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith('/') or not n.strip():
                    continue
                dest = os.path.join(d, os.path.basename(n))
                with open(dest, 'wb') as f:
                    f.write(z.read(n))
                out.append(dest)
        return ('expand', out, [], [])
    if ext == '.sqbpro':                       # DB Browser for SQLite : XML
        import xml.etree.ElementTree as ET
        try:
            root = ET.parse(path).getroot()
            blocks = [(s.get('name') or f'onglet {i+1}', s.text or '')
                      for i, s in enumerate(root.iter('sql'))]
            return ('text', [(os.path.basename(path) + ' : ' + n, c) for n, c in blocks], [], [])
        except Exception:
            pass
    if ext == '.ipynb':
        try:
            nb = json.load(open(path, encoding='utf-8'))
            blocks = []
            for i, c in enumerate(nb.get('cells', [])):
                src = ''.join(c.get('source', []))
                tag = 'cellule %d (%s)' % (i + 1, c.get('cell_type'))
                blocks.append((tag, src))
            return ('text', blocks, [], [])
        except Exception:
            pass
    if ext in IMG:
        return ('image', [], [path], [])
    if ext == '.pdf':
        return ('pdf', [], [], [path])
    t = read_text(path)
    if t is not None and ext not in {'.xlsx', '.docx', '.odt', '.pptx', '.7z', '.rar', '.gz', '.tar'}:
        return ('text', [(os.path.basename(path), t)], [], [])
    return ('binary', [(os.path.basename(path),
                        '<fichier binaire, %d octets, non inspectable>' % os.path.getsize(path))], [], [])


def items(files, workdir):
    """Entrees dans l'ordre d'origine : ('text',nom,contenu) | ('image',chemin) |
    ('pdf',chemin) | ('binaire',nom,taille). Les conteneurs sont developpes."""
    entries, stack = [], list(files)
    while stack:
        p = stack.pop(0)
        kind, blocks, ii, pp = degrade(p, workdir)
        if kind == 'expand':
            stack[:0] = blocks
            continue
        for name, text in blocks:
            entries.append(('text', name, text))
        for im in ii:
            entries.append(('image', im, None))
        for pd in pp:
            entries.append(('pdf', pd, None))
    return entries


def fenced(name, text, ext):
    lang = CODE.get(ext.lower(), '')
    text = (text or '').rstrip('\n')
    fence = '```'
    while fence in text:
        fence += '`'
    return '### %s\n\n%s%s\n%s\n%s\n' % (name, fence, lang, text, fence)


def to_md(student, entries, out):
    parts = [f'# {student.replace("_", " ")} \u2014 rendu\n']
    for kind, a, b in entries:
        if kind == 'text':
            parts.append(fenced(a, b, os.path.splitext(a)[1]))
        else:
            parts.append('### %s\n\n_(non textuel, voir le fichier joint original)_\n' % a)
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    return out


def to_pdf(student, entries, out):
    """PDF composable avec typst : texte en bloc verbatim, images incluses,
    pages de PDF incluses. doc.typ est pose dans le dossier courant afin que
    toutes les pieces (images, PDF inclus) restent sous la racine typst."""
    src = '.merge_doc.typ' 
    body = ['= Rendu ' + student.replace('_', ' '), '']
    for kind, a, b in entries:
        body += ['----', '==' + esc(str(a)) + '==', '']
        if kind == 'image':
            body.append('image("' + os.path.abspath(a) + '", width: 100%)')
        elif kind == 'pdf':
            body.append('include "' + os.path.abspath(a) + '"')
        elif kind == 'binaire':
            body.append('_(contenu non textuel, ' + fmt_size(b) + ')_')
        else:
            body.append('block(breakable: true, inset: 8pt, stroke: 0.4pt +gray(65%), [')
            for line in (b or '').split('\n'):
                body.append(esc(line))
            body += [']))']
        body.append('')
    with open(src, 'w', encoding='utf-8') as f:
        f.write('\n'.join(body) + '\n')
    r = subprocess.run(['typst', 'compile', '--root', os.getcwd(), src, out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('typst a echoue :\n' + (r.stderr or r.stdout)[:2000])
    os.path.exists(src) and os.remove(src)
    return out


def esc(s):
    return (str(s).replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]')
            .replace('*', '\\*').replace('<', '\\<').replace('#', '\\#'))


def fmt_size(n):
    return ('%.0f Ko' % (n / 1024)) if n < 10 << 20 else ('%.1f Mo' % (n / 1048576))


def groups(files):
    g = {}
    for f in files:
        base = os.path.basename(f)
        key = base.split('__')[0] if '__' in base else os.path.splitext(base)[0]
        g.setdefault(key, []).append(f)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='*', default=None)
    ap.add_argument('--pattern', default='*__*')
    a = ap.parse_args()
    files = a.files or [f for f in sorted(glob.glob(a.pattern))
                        if '__' in f and not os.path.isdir(f)]
    work = os.path.abspath(tempfile.mkdtemp(prefix='.merge_work_', dir='.'))
    report = []
    for key, fs in sorted(groups(files).items()):
        entries = items(sorted(fs), work)
        graphical = any(k in ('image', 'pdf') for k, _a, _b in entries)
        if graphical:
            out, kind = to_pdf(key, entries, key + '.pdf'), 'pdf'
        elif len(entries) == 1 and str(entries[0][1]).endswith('.md'):
            out, kind = str(entries[0][1]), 'deja md'
        else:
            out, kind = to_md(key, entries, key + '.md'), 'md'
        report.append((key, len(fs), kind, out))
    shutil.rmtree(work, ignore_errors=True)
    for key, n, kind, out in report:
        print('%-28s %d fichier(s) -> %s  [%s]' % (key, n, out, kind))


if __name__ == '__main__':
    main()
