#!/usr/bin/env python3
"""
preflight.py
============
Check everything that can be checked WITHOUT calling a paid API.

    python preflight.py                # full check
    python preflight.py --profile gpt-4.1   # only what that profile needs

Run this after cloning, before spending anything. It answers "will the
notebooks run on my machine, and if not, what exactly is missing" -- which is
otherwise only discoverable by starting a run and watching it fail halfway.

Exit code is 0 if nothing is broken, 1 if any FAIL. WARNs do not fail the run:
they flag things that only matter for some profiles (Gurobi, Ollama).

No API calls. No money. Nothing is written except a temporary file used to
test that the run directory is writable. API keys are checked for presence and
shape only -- their values are never printed.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import platform
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
_counts = {OK: 0, WARN: 0, FAIL: 0}


def say(status: str, label: str, detail: str = "") -> None:
    _counts[status] += 1
    line = f"  [{status}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def head(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------- #
# 1. interpreter
# --------------------------------------------------------------------------- #
def check_python() -> None:
    head("1. Python")
    v = sys.version_info
    detail = f"{platform.python_version()} at {sys.executable}"
    if v >= (3, 10):
        say(OK, "interpreter", detail)
    else:
        say(FAIL, "interpreter too old -- need 3.10+", detail)

    # The single most common failure mode: a notebook kernel that is not this
    # interpreter, so `pip install` lands somewhere else entirely.
    say(OK, "note",
        "in a notebook run `import sys; print(sys.executable)` -- if it "
        "differs from the path above, install with %pip, not pip")


# --------------------------------------------------------------------------- #
# 2. packages
# --------------------------------------------------------------------------- #
CORE = [
    ("yaml", "PyYAML"), ("pandas", "pandas"), ("numpy", "numpy"),
    ("dotenv", "python-dotenv"), ("tiktoken", "tiktoken"),
    ("langchain", "langchain"), ("langchain_core", "langchain-core"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_community", "langchain-community"),
    ("openai", "openai"), ("faiss", "faiss-cpu"),
    # pandas imports this lazily, only when it actually opens a .xlsx, so a
    # missing openpyxl surfaces as a mid-run ImportError rather than at import
    # time. It was absent from requirements.txt until 2026-08-01 and this
    # check did not look for it, so the first person to clone the repo hit it
    # instead of us. Needed by run_all_*_obj.py, fix_refdata.py, and the
    # gpt-oss notebook.
    ("openpyxl", "openpyxl"),
]
# Not needed by the main pipeline; a WARN here is not a blocker.
OPTIONAL = [
    ("gurobipy", "gurobipy"),        # score_runs.py and the label scripts
    ("nbformat", "nbformat"),
    ("matplotlib", "matplotlib"),    # run_all_Generate_Label_Large_Scale_Or
]


def _pinned() -> dict:
    req = HERE / "requirements.txt"
    if not req.exists():
        return {}
    out = {}
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Most of the file is `name==version`, but not all of it: entries added
        # later use `>=` on purpose. Matching only `==` silently treated those
        # as undeclared, which is the same class of blind spot this check
        # exists to catch.
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|~=|>|<=|<)?\s*([^\s;#]*)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(3) if m.group(2) == "==" else None
    return out


def check_packages() -> None:
    head("2. Packages")
    pins = _pinned()
    missing = []
    for mod, dist in CORE + OPTIONAL:
        optional = (mod, dist) in OPTIONAL
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            missing.append(dist)
            say(WARN if optional else FAIL, f"{dist} not importable",
                f"{type(e).__name__}: {e}")
            continue
        got = getattr(m, "__version__", None)
        want = pins.get(dist.lower())
        if want and got and got != want:
            say(WARN, f"{dist} {got}",
                f"requirements.txt pins {want} -- results may not match ours")
        else:
            say(OK, f"{dist} {got or ''}".rstrip())
    if missing:
        say(FAIL, "install the missing packages",
            "pip install -r requirements.txt   (in a notebook: %pip install ...)")

    _check_undeclared_imports(pins)


# Providers `leanopt_exp.build_llm` / `build_embeddings` import inside a branch,
# reached only when exp_config.yaml selects that provider. None of the current
# profiles do, so leaving them out of requirements.txt is deliberate, not an
# omission -- install them only if you add such a profile.
_OPTIONAL_PROVIDERS = {"langchain_anthropic", "langchain_huggingface"}

# Import name -> pip name, where they differ.
_PIP_NAME = {
    "yaml": "pyyaml", "dotenv": "python-dotenv", "faiss": "faiss-cpu",
    "sklearn": "scikit-learn", "PIL": "pillow", "bs4": "beautifulsoup4",
    "cv2": "opencv-python",
}


def _check_undeclared_imports(pins: dict) -> None:
    """Flag third-party imports that requirements.txt does not declare.

    The hand-maintained CORE list above only catches packages someone thought
    to add. requirements.txt was generated by `pip freeze` on a machine that
    already had openpyxl from elsewhere, so it was never listed -- and the gap
    stayed invisible until someone cloned the repo fresh and hit
    "Missing optional dependency 'openpyxl'" partway through a run.

    This walks every .py and notebook code cell for imports and compares
    against requirements.txt, so the next omission is caught here rather than
    by whoever clones next. Imports inside a function body are reported too:
    lazy imports are exactly the ones that fail late.
    """
    import sys as _sys

    declared = set(pins)
    stdlib = getattr(_sys, "stdlib_module_names", set())
    local = {p.stem for p in HERE.glob("*.py")}
    seen: dict[str, set[str]] = {}

    def walk(src: str, where: str) -> None:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            else:
                continue
            for n in names:
                seen.setdefault(n.split(".")[0], set()).add(where)

    for p in sorted(HERE.glob("*.py")):
        walk(p.read_text(encoding="utf-8", errors="ignore"), p.name)
    for p in sorted(HERE.glob("*.ipynb")):
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            continue
        for c in nb.get("cells", []):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            if not src.strip().startswith(("!", "%")):
                walk(src, p.name)

    undeclared = []
    for mod, where in sorted(seen.items()):
        if mod in stdlib or mod in local or mod.startswith("_"):
            continue
        if mod in _OPTIONAL_PROVIDERS:
            continue
        pip = _PIP_NAME.get(mod, mod).lower().replace("_", "-")
        if pip in declared:
            continue
        # Importable anyway? Then it is only a documentation gap, not a break.
        try:
            importlib.import_module(mod)
            installed = True
        except Exception:                                 # noqa: BLE001
            installed = False
        undeclared.append((pip, sorted(where), installed))

    if not undeclared:
        say(OK, "every third-party import is declared in requirements.txt")
        return
    for pip, where, installed in undeclared:
        say(WARN if installed else FAIL,
            f"{pip} is imported but not in requirements.txt"
            f"{'' if installed else ' -- and not installed'}",
            f"used by {', '.join(where)[:70]}")


# --------------------------------------------------------------------------- #
# 3. configuration
# --------------------------------------------------------------------------- #
COMBOS = [
    ("gpt-4.1", "LEAN-LLM-OPT", "Large-Scale-OR"),
    ("gpt-4.1", "LEAN-LLM-OPT", "Air-NRM-CA"),
    ("gpt-oss-20b", "LEAN-LLM-OPT", "Large-Scale-OR"),
    ("gpt-oss-20b", "LEAN-LLM-OPT", "Air-NRM-CA"),
    ("gpt-4.1", "Abl-RAGOnly", "Air-NRM-CA"),
    ("gpt-4.1", "Abl-FewShotOnly", "Air-NRM-CA"),
    ("gpt-oss-20b", "Base-SingleCall", "NL4OPT"),
]


def check_config(only_profile: str | None):
    head("3. Configuration (exp_config.yaml)")
    cfg_path = HERE / "exp_config.yaml"
    if not cfg_path.exists():
        say(FAIL, "exp_config.yaml missing", f"expected at {cfg_path}")
        return None
    try:
        import leanopt_exp as lx
    except Exception as e:
        say(FAIL, "cannot import leanopt_exp", f"{type(e).__name__}: {e}")
        return None

    last = None
    for prof, method, ds in COMBOS:
        if only_profile and prof != only_profile:
            continue
        try:
            last = lx.load_config(str(cfg_path), model_profile=prof,
                                  method=method, dataset=ds, repeat_index=0)
            say(OK, f"{prof} / {method} / {ds}")
        except Exception as e:
            say(FAIL, f"{prof} / {method} / {ds}", f"{type(e).__name__}: {e}")

    # dataset files
    import yaml
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for name, rel in (raw.get("datasets") or {}).items():
        if rel is None:
            continue
        say(OK if (HERE / rel).exists() else FAIL, f"dataset {name}", rel)
    return last


# --------------------------------------------------------------------------- #
# 4. notebooks
# --------------------------------------------------------------------------- #
PATH_RE = re.compile(r"""['"]((?:Test_Dataset|Large_Scale_Or_Files|Results)/[^'"]+)['"]""")


def check_notebooks() -> None:
    head("4. Notebooks")
    orig = HERE / "original_notebooks"
    if not orig.is_dir():
        say(FAIL, "original_notebooks/ missing",
            "it identifies which root notebooks belong to the pipeline")
        return
    names = sorted(p.name for p in orig.glob("*.ipynb"))
    if not names:
        say(FAIL, "original_notebooks/ has no .ipynb")
        return

    missing_paths = set()
    for name in names:
        nb_path = HERE / name
        if not nb_path.exists():
            say(FAIL, f"{name} missing from repository root")
            continue
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
        except Exception as e:
            say(FAIL, f"{name} is not valid JSON", str(e))
            continue

        bad_cells, refs = [], set()
        for i, c in enumerate(nb.get("cells", [])):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            stripped = src.strip()
            if not stripped.startswith(("!", "%")):
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    bad_cells.append(f"cell {i}: {e.msg} (line {e.lineno})")
            refs.update(PATH_RE.findall(src))

        gone = {r for r in refs
                if not (HERE / r).exists() and not r.endswith("information.csv")}
        missing_paths |= gone

        if bad_cells:
            say(FAIL, f"{name} has cells that do not parse", "; ".join(bad_cells[:3]))
        elif gone:
            say(FAIL, f"{name} references missing files", "; ".join(sorted(gone)[:3]))
        else:
            say(OK, f"{name}", f"{len(refs)} data paths, all present")

    if missing_paths:
        say(FAIL, f"{len(missing_paths)} referenced data file(s) do not exist",
            "the notebooks will fail at the read, not at the API call")


# --------------------------------------------------------------------------- #
# 4b. LaTeX eaten by Python's escape handling
# --------------------------------------------------------------------------- #
# LaTeX written in a plain (non-raw) string literal loses any command whose
# first letter happens to be a valid Python escape:
#
#     "\forall"  ->  formfeed + "orall"
#     "\begin"   ->  backspace + "egin"
#     "\text"    ->  tab       + "ext"
#
# Unknown escapes like "\sum" survive untouched, which is why the problem is
# easy to miss: most of the LaTeX looks fine. The affected strings are prompt
# text, so the model receives control characters where a command should be.
#
# Python only *warns* about this from 3.12 on, but the corruption itself is
# version-independent -- it happens on 3.11 too, silently.
_LATEX_CMD = re.compile(r"\\(?:sum|cdot|geq|quad|max|min|end|frac|ge|le|times|in)\b")
_EATEN = {
    "\x07": r"\a  (\alpha …)", "\x08": r"\b  (\begin …)",
    "\x0c": r"\f  (\forall, \frac …)", "\x0b": r"\v  (\vec …)",
    "\t": r"\t  (\text, \times …)",
}


def check_latex_escapes() -> None:
    head("4b. LaTeX in prompt strings")
    orig = HERE / "original_notebooks"
    names = sorted(p.name for p in orig.glob("*.ipynb")) if orig.is_dir() else []
    total, per_file = 0, {}

    for name in names:
        p = HERE / name
        if not p.exists():
            continue
        try:
            nb = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n = 0
        for c in nb.get("cells", []):
            if c.get("cell_type") != "code":
                continue
            src = "".join(c.get("source", []))
            if src.strip().startswith(("!", "%")):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant)
                        and isinstance(node.value, str)):
                    continue
                v = node.value
                if not _LATEX_CMD.search(v):
                    continue           # not a LaTeX string, tabs are fine there
                for ch in _EATEN:
                    n += v.count(ch)
        if n:
            per_file[name] = n
            total += n

    if not total:
        say(OK, "no LaTeX commands lost to escape handling")
        return
    say(WARN, f"{total} LaTeX command(s) corrupted by string escapes",
        "these strings are sent to the model as prompt text")
    for name, n in sorted(per_file.items(), key=lambda kv: -kv[1]):
        say(WARN, f"  {name}", f"{n} occurrence(s)")
    say(WARN, "  how to read this",
        r"e.g. \forall became a formfeed character plus 'orall'. "
        r"Fix by making those literals raw strings (r'...') or doubling "
        r"the backslashes. Known issue -- it predates the instrumentation "
        r"refactor and is present in original_notebooks/ too.")


# --------------------------------------------------------------------------- #
# 5. credentials  (presence and shape only -- values are never printed)
# --------------------------------------------------------------------------- #
def check_keys(cfg) -> None:
    head("5. Credentials")
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        say(OK if (HERE / ".env").exists() else WARN, ".env",
            "loaded" if (HERE / ".env").exists()
            else "not found -- run `python set_key.py OPENAI_API_KEY`")
    except Exception:
        say(WARN, "python-dotenv unavailable", "keys must come from the shell then")

    for var in ("OPENAI_API_KEY",):
        v = (os.environ.get(var) or "").strip()
        if not v:
            say(WARN, f"{var} not set",
                "only needed for the OpenAI profiles; "
                "run `python set_key.py OPENAI_API_KEY`")
        elif len(v) < 20 or v.endswith("..."):
            say(FAIL, f"{var} looks like a placeholder",
                "it was probably copied from .env.example unchanged")
        else:
            say(OK, f"{var} present", f"{len(v)} chars, value not shown")

    if (HERE / ".env").exists():
        gi = (HERE / ".gitignore")
        ignored = gi.exists() and ".env" in gi.read_text(encoding="utf-8")
        say(OK if ignored else FAIL, ".env excluded from git",
            "" if ignored else "add `.env` to .gitignore before committing!")


# --------------------------------------------------------------------------- #
# 6. solvers and local model servers
# --------------------------------------------------------------------------- #
def check_backends() -> None:
    head("6. Solver and local model server")
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        gp.Model(env=env).dispose()
        env.dispose()
        say(OK, "Gurobi license valid")
    except ImportError:
        say(WARN, "gurobipy not installed",
            "needed by score_runs.py and the label scripts, not by the notebooks")
    except Exception as e:
        say(WARN, "Gurobi license not usable",
            f"{type(e).__name__}: {str(e)[:120]}")

    import socket
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", 11434))
        say(OK, "Ollama reachable on :11434")
    except Exception:
        say(WARN, "Ollama not reachable on :11434",
            "only needed for the gpt-oss-20b profile")
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# 7. writability
# --------------------------------------------------------------------------- #
def check_writable() -> None:
    head("7. Output directory")
    runs = HERE / "runs"
    try:
        runs.mkdir(exist_ok=True)
        probe = runs / ".preflight_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        say(OK, "runs/ writable")
    except Exception as e:
        say(FAIL, "cannot write to runs/", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--profile", default=None,
                    help="only check this model profile's combinations")
    args = ap.parse_args()

    print(f"preflight -- {HERE}")
    check_python()
    check_packages()
    cfg = check_config(args.profile)
    check_notebooks()
    check_latex_escapes()
    check_keys(cfg)
    check_backends()
    check_writable()

    print("\n" + "=" * 60)
    print(f"  {_counts[OK]} OK   {_counts[WARN]} WARN   {_counts[FAIL]} FAIL")
    if _counts[FAIL]:
        print("\n  Fix the FAILs before running anything that costs money.")
        return 1
    if _counts[WARN]:
        print("\n  No blockers. WARNs are fine as long as they are about "
              "backends this profile does not use.")
    else:
        print("\n  All clear.")
    print("\n  Next, still free:")
    print("    python run_baseline.py --dataset Large-Scale-OR "
          "--profile gpt-4.1 --rows 15,45,81 --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
