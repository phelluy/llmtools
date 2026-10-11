#!/usr/bin/env python3
"""Serveur MCP exposant l'exécution de code Python, sandboxé dans un dossier.

Remplace mcp-python-interpreter (pin mcp<2) par un serveur FastMCP natif au SDK
MCP 2.x. Le code s'exécute dans un subprocess (pas in-process) avec un timeout
wall-clock et des limites de ressources (CPU, taille de fichier, mémoire sur
Linux).

Outils exposés :
  - run_python_code(code)    : exécute du code inline via un fichier temporaire
  - run_python_file(path)    : exécute un fichier .py du sandbox
  - list_sandbox_files()     : liste les fichiers et dossiers du sandbox
  - read_sandbox_file(path)  : lit le contenu d'un fichier texte du sandbox
  - run_shell_command(cmd, cwd, timeout) : exécute une commande shell (bash -c)

Lancé via uvx :
  uvx --with fastmcp --with sympy --with numpy ... \
      python mcp_python_server.py --dir <sandbox>
"""
import argparse
import os
import platform
import signal
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
MAX_OUTPUT_CHARS = 15000   # Seuil de tronquage pour protéger le contexte du LLM
SHELL_TIMEOUT = 120        # wall-clock par défaut pour run_shell_command (s)
SHELL_MAX_TIMEOUT = 600    # plafond du paramètre timeout (s)

mcp = FastMCP("python")
SANDBOX = Path(".").resolve()
# Python de l'env uvx (a les libs --with : sympy, numpy, ...). Utilisé par les
# deux outils run_python_code et run_python_file — un seul interpréteur.
SERVER_PY = sys.executable


def _limits(cpu_seconds: int = CPU):
    """Limites de ressources appliquées dans l'enfant avant exec (Unix)."""
    import resource
    for lim, val in (
        (getattr(resource, "RLIMIT_CPU", None), cpu_seconds),
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


def _truncate_output(text: str) -> str:
    """Tronque une chaîne si elle dépasse MAX_OUTPUT_CHARS."""
    if len(text) > MAX_OUTPUT_CHARS:
        excess = len(text) - MAX_OUTPUT_CHARS
        return (
            text[:MAX_OUTPUT_CHARS]
            + f"\n\n... [Sortie tronquée : {excess} caractères supplémentaires omis]"
        )
    return text


def _resolve_sandbox_path(path: str) -> Path:
    """Résout un chemin relatif au sandbox et vérifie le confinement."""
    target = (SANDBOX / path).resolve()
    if SANDBOX not in target.parents and target != SANDBOX:
        raise ValueError(f"chemin '{path}' hors du sandbox '{SANDBOX}'")
    return target


def _to_str(x) -> str:
    """TimeoutExpired.stdout/stderr : bytes, str ou None selon le mode de capture."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return x or ""


def _run(code_path: Path, py: str) -> str:
    try:
        proc = subprocess.run(
            [py, str(code_path)],
            cwd=str(SANDBOX),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            start_new_session=True,
            preexec_fn=_limits,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        out = _to_str(exc.stdout)
        err = _to_str(exc.stderr)
        msg = f"Erreur : le temps d'exécution a dépassé la limite autorisée ({TIMEOUT}s)."
        if out or err:
            msg += "\n\nSortie partielle avant interruption :\n" + out
            if err:
                msg += ("\n--- stderr ---\n" + err) if out else f"--- stderr ---\n{err}"
        return _truncate_output(msg)
    except Exception as e:
        return f"Erreur d'exécution du subprocess : {e}"

    out = _to_str(proc.stdout)
    err = _to_str(proc.stderr)
    if err:
        out += ("\n--- stderr ---\n" + err) if out else err
    if not out:
        return "(no output)"
    return _truncate_output(out)


@mcp.tool
def run_python_code(code: str) -> str:
    """Exécute du code Python inline dans un subprocess et renvoie stdout+stderr.
    Le code est écrit dans un fichier temporaire du sandbox puis exécuté via le
    python du serveur (env uvx, avec les libs --with), avec timeout wall-clock
    et limites CPU/fichier/mémoire.
    """
    with tempfile.NamedTemporaryFile(
        "w", prefix=".tmp_", suffix=".py", dir=str(SANDBOX), delete=False
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
    python du serveur (env uvx, avec les libs --with).
    """
    target = _resolve_sandbox_path(path)
    if target.is_dir():
        raise IsADirectoryError(f"Le chemin spécifié est un dossier, pas un fichier : {path}")
    if not target.is_file():
        raise FileNotFoundError(f"Fichier introuvable dans le sandbox : {path}")
    return _run(target, SERVER_PY)


@mcp.tool
def list_sandbox_files() -> str:
    """Liste les fichiers et dossiers présents dans le sandbox."""
    if not SANDBOX.exists():
        return f"Dossier sandbox introuvable : {SANDBOX}"
    entries = []
    for p in sorted(SANDBOX.rglob("*")):
        rel = p.relative_to(SANDBOX)
        # Ignorer fichiers et dossiers cachés
        if any(part.startswith(".") for part in rel.parts):
            continue
        # Ignorer fichiers et dossiers temporaires (ex: tmp*.py)
        if any(part.startswith("tmp") for part in rel.parts):
            continue
        if p.is_dir():
            entries.append(f"[dossier] {rel}")
        else:
            entries.append(f"[fichier] {rel} ({p.stat().st_size} octets)")
    if not entries:
        return "Le sandbox est vide."
    return "\n".join(entries)


@mcp.tool
def read_sandbox_file(path: str) -> str:
    """Lit le contenu d'un fichier texte présent dans le sandbox.
    Le chemin doit être confiné dans le sandbox. Le contenu est tronqué s'il
    dépasse le seuil de sécurité.
    """
    target = _resolve_sandbox_path(path)
    if target.is_dir():
        raise IsADirectoryError(f"Le chemin spécifié est un dossier, pas un fichier : {path}")
    if not target.is_file():
        raise FileNotFoundError(f"Fichier introuvable dans le sandbox : {path}")
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {e}"
    return _truncate_output(content)


@mcp.tool
def run_shell_command(cmd: str, cwd: str = "", timeout: int = 0) -> str:
    """Exécute une commande shell (bash -c) dans un dossier de travail et
    renvoie stdout+stderr ainsi que le code retour.

    Conçu pour un usage "harnais de vibe coding" : la commande est une chaîne
    shell complète (pipes, &&, redirections, variables d'environnement...).

    - cwd : dossier de travail de la commande. ~ et $VAR sont développés.
      Vide → dossier sandbox par défaut. Un chemin absolu quelconque est
      accepté, y compris hors du sandbox (interventions ponctuelles :
      configuration d'un logiciel, etc.) — c'est un choix assumé.
    - timeout : limite wall-clock en secondes (0 → défaut 120 s, plafond 600 s).
      Au dépassement, tout le groupe de processus est tué (SIGKILL).
    - Limites héritées du serveur : CPU ~= timeout, taille fichier écrite 50 MB,
      sortie tronquée à 15000 caractères.
    """
    if not cmd.strip():
        return "Erreur : commande vide."
    if timeout <= 0:
        timeout = SHELL_TIMEOUT
    timeout = min(timeout, SHELL_MAX_TIMEOUT)
    if cwd:
        workdir = Path(os.path.expandvars(os.path.expanduser(cwd))).resolve()
        if not workdir.is_dir():
            return f"Erreur : le dossier de travail n'existe pas : {workdir}"
    else:
        workdir = SANDBOX

    shell = "/bin/bash" if Path("/bin/bash").exists() else "/bin/sh"
    try:
        proc = subprocess.Popen(
            [shell, "-c", cmd],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=lambda: _limits(timeout),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception as e:
        return f"Erreur au lancement de la commande : {e}"

    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Tuer tout le groupe (bash + descendants), pas seulement bash.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        msg = (
            f"Erreur : la commande a dépassé la limite de {timeout}s "
            "(groupe de processus tué)."
        )
        if out or err:
            msg += "\n\nSortie partielle avant interruption :\n" + (out or "")
            if err:
                msg += "\n--- stderr ---\n" + err
        return _truncate_output(msg)

    output = out or ""
    if err:
        output += ("\n--- stderr ---\n" + err) if output else err
    header = f"[cwd={workdir} | code={proc.returncode}]"
    if not output:
        return _truncate_output(f"{header}\n(no output)")
    return _truncate_output(f"{header}\n{output}")


def main():
    global SANDBOX
    p = argparse.ArgumentParser(description="Serveur MCP d'exécution Python (FastMCP)")
    p.add_argument("--dir", required=True, help="dossier sandbox")
    args, _ = p.parse_known_args()
    SANDBOX = Path(args.dir).resolve()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
