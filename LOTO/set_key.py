#!/usr/bin/env python3
"""
set_key.py
==========
Write an API key into the .env file without having to find or open the file.
The key is typed into a hidden prompt and never echoed or printed back.

    python set_key.py OPENROUTER_API_KEY
    python set_key.py OPENAI_API_KEY
    python set_key.py --show          # list which keys are set (values masked)

If the variable already exists in .env its line is replaced; other lines are
left untouched.
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path

ENV = Path(__file__).parent / ".env"
KNOWN = ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY",
         "ANTHROPIC_API_KEY"]


def read_lines() -> list[str]:
    if not ENV.exists():
        return []
    return ENV.read_text(encoding="utf-8").splitlines()


def parse(lines: list[str]) -> dict[str, str]:
    out = {}
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def mask(v: str) -> str:
    if not v or v.endswith("...") or v in ("sk-...", "sk-or-v1-..."):
        return "(placeholder, not a real key)"
    return f"***{v[-4:]}  ({len(v)} chars)"


def show():
    lines = read_lines()
    if not lines:
        print(f"{ENV} does not exist yet")
        return
    have = parse(lines)
    print(f"{ENV}")
    for k in KNOWN:
        print(f"  {k:22} {mask(have[k]) if k in have else '-- not set --'}")
    extra = [k for k in have if k not in KNOWN]
    if extra:
        print(f"  other variables: {extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("var", nargs="?", help="variable name, e.g. OPENROUTER_API_KEY")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.show or not args.var:
        show()
        return 0

    var = args.var.strip().upper()
    value = getpass.getpass(f"paste {var} (input is hidden, then press Enter): ").strip()
    if not value:
        print("nothing entered, .env unchanged")
        return 1
    if " " in value or "\n" in value:
        print("the value contains a space or newline -- that is almost certainly "
              "a copy-paste error; .env unchanged")
        return 1

    lines = read_lines()
    pat = re.compile(rf"^\s*{re.escape(var)}\s*=")
    replaced = False
    for i, ln in enumerate(lines):
        if pat.match(ln):
            lines[i] = f"{var}={value}"
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{var}={value}")

    ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ENV.chmod(0o600)                     # readable only by you
    print(f"{'replaced' if replaced else 'added'} {var} in {ENV.name} "
          f"-> {mask(value)}")
    print("now restart the notebook kernel and re-run the configuration cell")
    return 0


if __name__ == "__main__":
    sys.exit(main())
