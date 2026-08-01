#!/usr/bin/env python3
"""
switch_profile.py
=================
Flip which model_profile a notebook's configuration cell uses, without
hand-editing JSON.

    python switch_profile.py --list
    python switch_profile.py openrouter-gpt-4.1
    python switch_profile.py gpt-4.1 --only Large-scale-or
    python switch_profile.py ollama-small --only gpt_oss

By default only notebooks currently on an OpenAI-family profile are switched,
so the local gpt-oss runs are left alone. Use --force to switch anything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
PROFILE_RE = re.compile(r"(model_profile\s*=\s*)(['\"])([^'\"]+)\2")

# profiles that are interchangeable without changing anything else
OPENAI_FAMILY = {"gpt-4.1", "gpt-5.2", "gemini-3-pro", "openrouter-gpt-4.1"}


def notebooks():
    """The instrumented notebooks in the repository root.

    Identified by name against ``original_notebooks/``: every file in there has
    exactly one instrumented counterpart at the root. Deriving the list this
    way means adding or removing a notebook needs no edit here, and it keeps
    utilities that are *not* part of the pipeline
    (``run_all_Generate_Label_Large_Scale_Or.ipynb``,
    ``uflp_txt_to_gurobipy_and_obj.ipynb``) out of the selection.
    """
    orig_dir = HERE / "original_notebooks"
    if not orig_dir.is_dir():
        raise SystemExit(
            f"cannot find {orig_dir} -- it is what identifies the instrumented "
            f"notebooks. Run this script from the repository root.")
    names = {p.name for p in orig_dir.glob("*.ipynb")}
    return sorted(p for p in HERE.glob("*.ipynb") if p.name in names)


def current_profile(nb: dict):
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        m = PROFILE_RE.search("".join(c["source"]))
        if m:
            return m.group(3)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", nargs="?", help="target model_profile")
    ap.add_argument("--only", default=None,
                    help="substring filter on the notebook file name")
    ap.add_argument("--list", action="store_true",
                    help="show each notebook's current profile")
    ap.add_argument("--force", action="store_true",
                    help="also switch notebooks on a local profile")
    args = ap.parse_args()

    files = [f for f in notebooks() if not args.only or args.only in f.name]
    if not files:
        print("no matching notebook")
        return 1

    if args.list or not args.profile:
        width = max(len(f.name) for f in files)
        for f in files:
            nb = json.loads(f.read_text(encoding="utf-8"))
            print(f"  {f.name:{width}}  {current_profile(nb)}")
        return 0

    # sanity: the profile has to exist in the config
    import yaml
    cfg = yaml.safe_load((HERE / "exp_config.yaml").read_text(encoding="utf-8"))
    known = list(cfg["model_profiles"])
    if args.profile not in known:
        print(f"unknown profile {args.profile!r}; available: {known}")
        return 1

    changed = 0
    for f in files:
        nb = json.loads(f.read_text(encoding="utf-8"))
        cur = current_profile(nb)
        if cur is None:
            print(f"  skip {f.name}: no model_profile found")
            continue
        if cur == args.profile:
            print(f"  ok   {f.name}: already {cur}")
            continue
        if cur not in OPENAI_FAMILY and not args.force:
            print(f"  skip {f.name}: on local profile {cur!r} (use --force)")
            continue

        hit = 0
        for c in nb["cells"]:
            if c["cell_type"] != "code":
                continue
            s = "".join(c["source"])
            s2, k = PROFILE_RE.subn(
                lambda m: f"{m.group(1)}{m.group(2)}{args.profile}{m.group(2)}", s)
            if k:
                hit += k
                c["source"] = s2.splitlines(keepends=True)
        f.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  set  {f.name}: {cur} -> {args.profile}  ({hit} site(s))")
        changed += 1

    print(f"\n{changed} notebook(s) changed. "
          f"Close and reopen them in the editor, then restart the kernel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
