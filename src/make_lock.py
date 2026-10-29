"""
Write requirements-lock.txt from the CURRENTLY INSTALLED environment. Installs/upgrades nothing.

    python src\\make_lock.py            # create requirements-lock.txt (refuses to overwrite without --force)
    python src\\make_lock.py --check    # compare the installed environment with an existing lock file
"""
from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(lines) -> dict:
    """{normalised name: 'name==version'} for exact pins; other forms (e.g. 'pkg @ file:...') are returned under '__other__'."""
    pins, other = {}, []
    for raw in lines:
        l = raw.strip()
        if not l or l.startswith("#"):
            continue
        if "==" in l and "@" not in l:
            pins[norm(l.split("==")[0])] = l
        else:
            other.append(l)
    pins["__other__"] = other
    return pins


def missing_requirements(req_lines, pins) -> list:
    out = []
    for l in req_lines:
        l = l.strip()
        if l and not l.startswith("#"):
            name = re.split(r"[<>=!~\[;\s]", l, maxsplit=1)[0]
            if norm(name) not in pins:
                out.append(name)
    return out


def freeze() -> list:
    r = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"], capture_output=True, text=True, check=True)
    return r.stdout.splitlines()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "requirements-lock.txt"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare installed environment to the existing lock; change nothing")
    a = ap.parse_args(argv)
    out = Path(a.out)
    installed = parse_pins(freeze())

    if a.check:
        if not out.exists():
            print(f"No lock file at {out}")
            return 2
        locked = parse_pins(out.read_text(encoding="utf-8").splitlines())
        diffs = [f"{locked[k]}  (installed: {installed.get(k, 'NOT INSTALLED')})" for k in locked if k != "__other__" and installed.get(k) != locked[k]]
        extra = [installed[k] for k in installed if k != "__other__" and k not in locked]
        print("Lock matches installed environment." if not diffs and not extra else "DIFFERENCES FOUND")
        for d in diffs:
            print("  differs:", d)
        for e in extra:
            print("  installed but not in lock:", e)
        return 0 if not diffs and not extra else 1

    req_file = ROOT / "requirements.txt"
    miss = missing_requirements(req_file.read_text().split() if req_file.exists() else [], installed)
    if miss:
        print(f"ERROR: not installed in this environment: {miss}. Activate the project .venv first.")
        return 1
    if out.exists() and not a.force:
        print(f"Refusing to overwrite {out}. Use --force.")
        return 1
    pins = sorted((v for k, v in installed.items() if k != "__other__"), key=str.lower)
    header = [f"# Lock file generated from the installed environment on {datetime.now():%Y-%m-%d %H:%M}",
              f"# Python {platform.python_version()} on {platform.system()}; nothing was installed or upgraded to create it.",
              "# Restore with: python -m venv .venv ; pip install -r requirements-lock.txt", "#"]
    other = installed["__other__"]
    out.write_text("\n".join(header + pins + ([f"# NOT exact pins, review manually: {o}" for o in other])) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(pins)} pinned packages)")
    chk = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    print("pip check:", chk.stdout.strip() or "ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
