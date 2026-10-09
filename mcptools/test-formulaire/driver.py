"""Test end-to-end du serveur MCP Playwright : remplissage d'un formulaire (texte + checkboxes + select).

Pilote le serveur en JSON-RPC sur stdio, comme le ferait un client MCP (harnais IA) :
navigate -> snapshot (refs) -> browser_fill_form -> browser_click (Envoyer) -> verification.

Usage : ./run.sh (sert form.html sur le port 8765 puis lance ce driver).
NB : la version @latest de @playwright/mcp exige, pour browser_fill_form, des champs
{target, name, type, value} et, pour browser_click, l'argument target (et non ref).
"""
import json, re, subprocess

FORM_URL = "http://127.0.0.1:8765/form.html"
EXPECTED = {"nom": "Jean Dupont", "email": "jean.dupont@example.com",
            "newsletter": True, "cgu": True, "taille": "M"}

p = subprocess.Popen(
    ["npx", "-y", "@playwright/mcp@latest", "--browser", "chrome", "--headless", "--isolated"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1)

_ids = iter(range(1, 100))

def send(method, params=None, notify=False):
    msg = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    if not notify:
        msg["id"] = next(_ids)
    p.stdin.write(json.dumps(msg) + "\n")
    p.stdin.flush()

def recv():
    while True:
        line = p.stdout.readline()
        if not line:
            raise EOFError("serveur fermé")
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if "id" in m:
            if "error" in m:
                raise RuntimeError(m["error"])
            return m["result"]

def call(name, args):
    send("tools/call", {"name": name, "arguments": args})
    return recv()

def text_of(res):
    for c in res.get("content", []):
        if c.get("type") == "text":
            return c["text"]
    return ""

def find(items, kind, *keywords):
    for it_kind, label, ref in items:
        if it_kind == kind and all(k.lower() in label.lower() for k in keywords):
            return ref, label
    raise KeyError(f"{kind} {keywords} introuvable")

# --- 1. Poignee de main MCP ---
send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "test-driver", "version": "0"}})
info = recv()
print(f"[1] initialize OK : {info['serverInfo']['name']} {info['serverInfo']['version']}")
send("notifications/initialized", notify=True)

# --- 2. Navigation + snapshot d'accessibilite (les refs) ---
snap = text_of(call("browser_navigate", {"url": FORM_URL}))
print(f"[2] navigation OK : {[l for l in snap.splitlines() if 'Page Title' in l][0].strip()}")

items_snap = text_of(call("browser_snapshot", {}))
items = re.findall(r'- (textbox|checkbox|combobox) "([^"]+)" \[ref=(e\d+)\]', items_snap)
print(f"[3] snapshot : {len(items)} champs detectes -> {[(k, l, r) for k, l, r in items]}")

# --- 3. Remplissage du formulaire en un seul appel ---
ref_nom, _ = find(items, "textbox", "Nom")
ref_email, _ = find(items, "textbox", "Email")
ref_news, _ = find(items, "checkbox", "newsletter")
ref_cgu, _ = find(items, "checkbox", "CGU")
ref_taille, _ = find(items, "combobox", "Taille")
res_fill = text_of(call("browser_fill_form", {"fields": [
    {"target": ref_nom, "name": "Nom", "type": "textbox", "value": "Jean Dupont"},
    {"target": ref_email, "name": "Email", "type": "textbox", "value": "jean.dupont@example.com"},
    {"target": ref_news, "name": "newsletter", "type": "checkbox", "value": "true"},
    {"target": ref_cgu, "name": "cgu", "type": "checkbox", "value": "true"},
    {"target": ref_taille, "name": "taille", "type": "combobox", "value": "M"},
]}))
code = [l.strip() for l in res_fill.splitlines() if "await page" in l]
print(f"[4] fill_form OK, code Playwright execute :")
for c in code:
    print(f"      {c}")

# --- 4. Clic sur Envoyer ---
m = re.search(r'- button "Envoyer" \[ref=(e\d+)\]', items_snap)
res_click = call("browser_click", {"element": "bouton Envoyer", "target": m.group(1)})
print("[5] clic sur Envoyer OK, réponse complète :")
print(text_of(res_click).strip().replace("\n", "\n      "))

# --- 5. Verification : valeurs soumises dans la page ---
final = text_of(call("browser_snapshot", {}))
print("[6] snapshot final (extrait autour du résultat) :")
for l in final.splitlines():
    if "nom" in l or "res" in l.lower() or "Résultat" in l:
        print(f"      {l.strip()}")
m = re.search(r'generic \[ref=e\d+\]: ("?\{.*\}"?)', final)
if m:
    val = m.group(1)
    if val.startswith('"'):
        val = json.loads(val)
    got = json.loads(val)
else:
    got = None
print(f"[6] page apres soumission : {got}")
ok = got == EXPECTED
print(f"\n{'TEST REUSSI' if ok else 'TEST ECHEC'} : attendu {EXPECTED}")

p.terminate()
