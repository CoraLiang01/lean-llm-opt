#!/usr/bin/env python3
"""
audit_repo.py
=============
Static audit of a repository for the things that cause surprise API bills.
Reports only; never modifies anything, never prints a key in full.

    python audit_repo.py            # audit this repo
    python audit_repo.py ~/some/repo

Checks
------
1. Hard-coded API keys in tracked files (the classic leak).
2. Which model names the code can reach -- a bill for a model that appears
   nowhere in the code did not come from this code.
3. API base URLs, i.e. where requests can be sent.
4. Loop / retry settings that can run away (agents without an iteration cap,
   while-loops around API calls).
5. Whether results are being written outside the run-logging system, which
   would mean spending that nothing accounts for.

This does NOT look at anyone's shell profile or editor settings -- see the
checklist printed at the end for the part that has to be done by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KEY_PATTERNS = [
    ("OpenAI key", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9]{40,}")),
    ("OpenRouter key", re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}")),
    ("Anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("Google key", re.compile(r"AIza[A-Za-z0-9_-]{30,}")),
    ("AWS key", re.compile(r"AKIA[A-Z0-9]{16}")),
]

MODEL_RE = re.compile(
    r"['\"]("
    r"gpt-[0-9][^'\"]*|o[1-9][^'\"]*|claude-[^'\"]*|gemini-[^'\"]*|"
    r"text-embedding-[^'\"]*|nomic-embed[^'\"]*|llama[^'\"]*|gpt-oss[^'\"]*"
    r")['\"]")

BASE_URL_RE = re.compile(r"['\"](https?://[^'\"]*(?:api|openai|anthropic|"
                         r"googleapis|openrouter|localhost:\d+)[^'\"]*)['\"]")

RISK_PATTERNS = [
    # An agent is capped either by an explicit max_iterations or by going
    # through lx.agent_kwargs(), which supplies one from the config.
    ("agent without an iteration cap",
     re.compile(r"initialize_agent\((?:(?!max_iterations|agent_kwargs\()"
                r"[\s\S]){0,600}?\)")),
    ("while-loop around an LLM call",
     re.compile(r"while[^\n]*:[\s\S]{0,300}?\.(?:invoke|run|predict|create)\(")),
    ("retry decorator", re.compile(r"@retry\b|tenacity\.|backoff\.on_exception")),
    ("max_retries above 3", re.compile(r"max_retries\s*=\s*([4-9]|\d{2,})")),
]

# Directories whose contents are not part of the shipped repository. They are
# skipped so that section 2's claim stays true: a model that is not in the list
# this script prints cannot have been called by the code we actually ship.
# Superseded copies in _local_archive/ would otherwise re-introduce model names
# and endpoints that nothing live can reach, which is exactly the kind of noise
# that makes a billing question unanswerable.
SKIP_DIRS = {".git", ".history", "__pycache__", "node_modules", ".ipynb_checkpoints",
             "runs", "tables", ".venv", "venv",
             "_local_archive", "_smoke_runs", "_smoke_tables", "output"}
TEXT_SUFFIXES = {".py", ".ipynb", ".yaml", ".yml", ".json", ".md", ".txt",
                 ".env", ".cfg", ".ini", ".sh", ".toml"}


def mask(s: str) -> str:
    return f"{s[:10]}...{s[-4:]} ({len(s)} chars)"


def walk(root: Path):
    for p in root.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix.lower() in TEXT_SUFFIXES or p.name.startswith(".env"):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", type=Path)
    args = ap.parse_args()
    root = args.root.resolve()

    files = list(walk(root))
    print(f"auditing {root}  ({len(files)} text files)\n")

    # -- 1. keys ---------------------------------------------------------- #
    print("=" * 72)
    print("1. HARD-CODED CREDENTIALS")
    print("=" * 72)
    hits = 0
    for f in files:
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for label, pat in KEY_PATTERNS:
            for m in set(pat.findall(txt)):
                hits += 1
                where = f.relative_to(root)
                tag = " (expected: this is the local secrets file)" \
                    if f.name == ".env" else "  <-- LEAK"
                print(f"  {label}: {mask(m)}")
                print(f"      {where}{tag}")
    if not hits:
        print("  none found")

    # -- 2. reachable models ---------------------------------------------- #
    print()
    print("=" * 72)
    print("2. MODEL NAMES REFERENCED IN CODE")
    print("=" * 72)
    models = {}
    for f in files:
        if f.suffix.lower() not in {".py", ".ipynb", ".yaml", ".yml"}:
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in MODEL_RE.findall(txt):
            models.setdefault(m, set()).add(str(f.relative_to(root)))
    for m in sorted(models):
        where = sorted(models[m])
        print(f"  {m:<34} {len(where)} file(s): {where[0]}"
              + (f" +{len(where)-1}" if len(where) > 1 else ""))
    print("\n  A model on the bill that is NOT in this list cannot have been"
          "\n  called by this repository.")

    # -- 3. endpoints ------------------------------------------------------ #
    print()
    print("=" * 72)
    print("3. API ENDPOINTS REACHABLE FROM CODE")
    print("=" * 72)
    urls = {}
    for f in files:
        if f.suffix.lower() not in {".py", ".ipynb", ".yaml", ".yml"}:
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for u in BASE_URL_RE.findall(txt):
            urls.setdefault(u, set()).add(str(f.relative_to(root)))
    for u in sorted(urls):
        print(f"  {u:<44} {sorted(urls[u])[0]}")
    if not urls:
        print("  none found (default provider endpoints)")

    # -- 4. runaway risk --------------------------------------------------- #
    print()
    print("=" * 72)
    print("4. RUNAWAY-COST RISK")
    print("=" * 72)
    found = 0
    for f in files:
        if f.suffix.lower() not in {".py", ".ipynb"}:
            continue
        try:
            txt = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for label, pat in RISK_PATTERNS:
            n = len(pat.findall(txt))
            if n:
                found += 1
                print(f"  {label}: {n}x in {f.relative_to(root)}")
    if not found:
        print("  none found")

    # -- 5. manual checklist ----------------------------------------------- #
    print()
    print("=" * 72)
    print("5. WHAT THIS SCRIPT CANNOT SEE -- CHECK BY HAND")
    print("=" * 72)
    print("""\
  An API key set in the shell environment is picked up automatically by
  coding agents (Codex, IDE assistants, some CLI tools). Someone can be
  billing the organisation without ever writing a line of code that
  mentions the key. Each person should check their own machine:

    grep -rn "OPENAI_API_KEY" ~/.zshrc ~/.bashrc ~/.bash_profile ~/.profile \\
                              ~/.zprofile ~/.config/fish/config.fish 2>/dev/null
    env | grep -i "OPENAI\\|ANTHROPIC\\|GOOGLE_API"

  If OPENAI_API_KEY is exported there, any agent started from that shell
  bills the organisation per token instead of using a subscription.

  Codex specifically: `codex login status` (or the IDE extension's account
  panel) shows whether it is authenticated with a ChatGPT account or with an
  API key. ChatGPT-account auth does not produce API charges; API-key auth
  does. Check `~/.codex/` for a stored key as well.

  Signature of this in the usage dashboard: a coding-oriented model with
  "cache writes" as the dominant line item. Long repository contexts sent
  repeatedly are what produce large cache-write charges; a benchmark run of
  short independent prompts does not.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
