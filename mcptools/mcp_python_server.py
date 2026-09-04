#!/usr/bin/env python3
"""Serveur MCP exposant l'exécution de code Python, sandboxé dans un dossier.

Remplace mcp-python-interpreter (pin mcp<2) par un serveur FastMCP natif au SDK
MCP 2.x. Le code s'exécute dans un subprocess (pas in-process) avec un timeout
wall-clock et des limites de ressources (CPU, taille de fichier, mémoire sur
Linux).

Outils exposés (mêmes noms que mcp-python-interpreter) :
  - run_python_code(code) : exécute du code inline via un fichier temporaire
  - run_python_file(path)  : exécute un fichier .py du sandbox

Lancé via uvx :
  uvx --with fastmcp --with sympy --with numpy ... \
      python mcp_python_server.py --dir <sandbox> --python-path <venv python>
"""
import argparse
import platform
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from fastmcp import FastMCP

# Limites appliquées au subprocess enfant (avant exec, via preexec_fn).
TIMEOUT = 30        # wall-clock seconds
CPU = 25            # RLIMIT_CPU seconds
FSIZE = 50 * 1024 * 1024   # 50 MB max écriture fichier
MEM = 2 * 1024 * 1024 * 1024  # 2 GB, RLIMIT_AS (Linux seulement)

mcp = FastMCP("python")
SANDBOX = Path(".").resolve()
# SERVER_PY = python de l'env uvx (a les libs --with : sympy, numpy, ...).
# VENV_PY = python du venv utilisateur (pour run_python_file).
SERVER_PY = sys.executable
VENV_PY = sys.executable


def _limits():
    """Limites de ressources appliquées dans l'enfant avant exec (Unix)."""
    import resource
    for lim, val in (
        (getattr(resource, "RLIMIT_CPU", None), CPU),
        (getattr(resource, "RLIMIT_FSIZE", None), FSIZE),
    ):
        if lim is not None:
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
    # RLIMIT_AS n'est fiable que sur Linux : sur macOS l'espace d'adressage
    # virtuel de numpy/scipy est énorme et la limite tuerait des process légitimes.
    if platform.system() == "Linux":
        lim = getattr(__import__("resource"), "RLIMIT_AS", None)
        if lim is not None:
            try:
                __import__("resource").setrlimit(lim, (MEM, MEM))
            except (ValueError, OSError):
                pass


def _run(code_path: Path, py: str) -> str:
    proc = subprocess.run(
        [py, str(code_path)],
        cwd=str(SANDBOX),
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        start_new_session=True,
        preexec_fn=_limits,
    )
    out = proc.stdout
    if proc.stderr:
        out += ("\n--- stderr ---\n" + proc.stderr) if out else proc.stderr
    return out or "(no output)"


@mcp.tool
def run_python_code(code: str) -> str:
    """Exécute du code Python inline dans un subprocess et renvoie stdout+stderr.
    Le code est écrit dans un fichier temporaire du sandbox puis exécuté via le
    python du serveur (env uvx, avec les libs --with), avec timeout wall-clock
    et limites CPU/fichier/mémoire.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", dir=str(SANDBOX), delete=False
    ) as f:
        f.write(textwrap.dedent(code))
        tmp = Path(f.name)
    try:
        return _run(tmp, SERVER_PY)
    finally:
        tmp.unlink(missing_ok=True)


@mcp.tool
def run_python_file(path: str) -> str:
    """Exécute un fichier .py du sandbox dans un subprocess et renvoie
    stdout+stderr. Le chemin doit être à l'intérieur du sandbox. Utilise le
    python du venv utilisateur (--python-path).
    """
    target = (SANDBOX / path).resolve()
    if SANDBOX not in target.parents and target != SANDBOX:
        raise ValueError(f"chemin {path} hors du sandbox {SANDBOX}")
    return _run(target, VENV_PY)


def main():
    global SANDBOX, VENV_PY
    p = argparse.ArgumentParser(description="Serveur MCP d'exécution Python (FastMCP)")
    p.add_argument("--dir", required=True, help="dossier sandbox")
    p.add_argument("--python-path", required=True, help="python du venv pour run_python_file")
    args, _ = p.parse_known_args()
    SANDBOX = Path(args.dir).resolve()
    VENV_PY = args.python_path
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
