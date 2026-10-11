"""Test rapide de run_shell_command via mcp_python_server.py en JSON-RPC stdio."""
import json, subprocess

p = subprocess.Popen(
    ["uvx", "--with", "fastmcp", "python", "./mcp_python_server.py",
     "--dir", "./workdir/scripts"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1,
)
_ids = iter(range(1, 100))

def send(method, params=None, notify=False):
    msg = {"jsonrpc": "2.0", "method": method}
    msg["params"] = params if params else {}
    if not notify: msg["id"] = next(_ids)
    p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()

def recv():
    while True:
        line = p.stdout.readline()
        if not line: raise EOFError
        line = line.strip()
        if not line: continue
        try: m = json.loads(line)
        except ValueError: continue
        if "id" in m:
            if "error" in m: raise RuntimeError(m["error"])
            return m["result"]

def call(name, args):
    send("tools/call", {"name": name, "arguments": args})
    return recv()

def text_of(res):
    return "".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")

send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"}})
recv()
send("notifications/initialized", notify=True)
send("tools/list")
print("[tools]:", [t["name"] for t in recv()["tools"]])

print("\n[1] nominal /tmp:", text_of(call("run_shell_command",
      {"cmd": "echo bonjour && pwd", "cwd": "/tmp"})))

print("\n[2] ~ expansion:", text_of(call("run_shell_command",
      {"cmd": "ls", "cwd": "~/gitlab/llmtools"}))[:300])

print("\n[3] code retour:", text_of(call("run_shell_command",
      {"cmd": "echo ko >&2; exit 3", "cwd": "/tmp"})))

print("\n[4] cwd inexistant:", text_of(call("run_shell_command",
      {"cmd": "ls", "cwd": "/dossier/inexistant"})))

print("\n[5] sandbox par defaut:", text_of(call("run_shell_command", {"cmd": "pwd"})))

print("\n[6] timeout group kill:", text_of(call("run_shell_command",
      {"cmd": "sleep 60 & sleep 65", "timeout": 3})))

import time; time.sleep(1)
out = subprocess.run(["pgrep", "-fl", "sleep 6"], capture_output=True, text=True).stdout
print("\n[7] sleeps residuels apres kill:", repr(out.strip() or "(aucun)"))

r = text_of(call("run_shell_command", {"cmd": "seq 1 100000", "cwd": "/tmp"}))
print("\n[8] tronquage: longueur =", len(r), "| fin:", r[-120:].replace("\n", "\\n"))

print("\n[9] pipeline+env:", text_of(call("run_shell_command",
      {"cmd": "FOO=bar; echo $FOO | tr a-z A-Z", "cwd": "/tmp"})))

p.terminate()
