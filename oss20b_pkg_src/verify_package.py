#!/usr/bin/env python3
"""
verify_package.py
=================
Offline self-check for one task directory of the gpt-oss-20B run package.

    cd large_scale_or && python ../verify_package.py
    cd loto          && python ../verify_package.py

Calls no model and needs no API key: everything here is about whether the
package that arrived is the package that was sent, and whether this machine can
run it. Run it before spending GPU hours, not after.

What it does NOT prove: that Ollama honours `num_ctx`, that `reasoning: true`
separates the chain-of-thought from the answer, or that the model is any good.
Those only show up in the smoke run -- see the pass criteria in README.md.

Exit code is the number of FAILs, so it can gate a script.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path.cwd()
PKG_ROOT = Path(__file__).resolve().parent

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
_counts = {OK: 0, WARN: 0, FAIL: 0}


def say(level: str, msg: str, detail: str = "") -> None:
    _counts[level] += 1
    print(f"  [{level}] {msg}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")


def head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 0. which task directory is this
# --------------------------------------------------------------------------- #
IS_LOTO = (HERE / "run_loto_fold.py").exists()
IS_AIRNRM = (HERE / "LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_LargeScale.ipynb").exists()
TASK = "loto" if IS_LOTO else "air_nrm_ls" if IS_AIRNRM else "large_scale_or"
NOTEBOOK = {
    "loto": "LEAN_LLM_OPT_gpt_oss_20b_LOTO.ipynb",
    "air_nrm_ls": "LEAN_LLM_OPT_gpt_oss_20b_Air_NRM_LargeScale.ipynb",
    "large_scale_or": "LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb",
}[TASK]
DATASET, DATASET_ROWS = (
    (Path("Test_Dataset/Air_NRM/large_scale/query_CA.csv"), 15) if IS_AIRNRM
    else (Path("Test_Dataset/Large-scale-or/Large-scale-or-101.csv"), 101))

# Files the Air-NRM pipeline opens by fixed path rather than through a
# Dataset_address column. All five data tables come from large_scale/ --
# including flight_capacity.csv, which has no small-scale counterpart because
# every flight there is 187 seats and the number was hard-coded.
AIRNRM_FIXED_PATHS = [
    "Test_Dataset/Air_NRM/large_scale/v1.csv",
    "Test_Dataset/Air_NRM/large_scale/v2.csv",
    "Test_Dataset/Air_NRM/large_scale/od_demand.csv",
    "Test_Dataset/Air_NRM/large_scale/flight.csv",
    "Test_Dataset/Air_NRM/large_scale/flight_capacity.csv",
    "Test_Dataset/Air_NRM/large_scale/query_CA.csv",
    "Large_Scale_Or_Files/RefData.csv",
    # The large-scale exemplar. The small-scale one is written over the toy ODs
    # ('A','B'), which do not exist in this v1.csv.
    "Large_Scale_Or_Files/RAG_Example_SBLP_CA_LargeScale.csv",
    "Large_Scale_Or_Files/RAG_Example_SBLP_Flow.csv",
]

# RefData.csv carries one malformed Data_address cell -- three filenames run
# together into a single path. It predates this package and no code reads it
# (the row it belongs to is an Air-NRM reference, not a Large-Scale-OR one), so
# it is whitelisted rather than "fixed" into something that was never there.
KNOWN_BAD_PATHS = {"v1.csv/v2.csv/information_addB_2.csv"}

# Where the setup cells end -- the same rule run_notebook.py uses: the first
# cell that *calls* (rather than defines) the batch function.
BATCH_CALL_RE = re.compile(
    r"^(?!\s*def\s)\s*(?:\w+\s*(?:,\s*\w+\s*)*=\s*)?"
    r"(Batch_Process_Queries|run_test)\s*\(", re.M)


# --------------------------------------------------------------------------- #
# 1. interpreter and dependencies
# --------------------------------------------------------------------------- #
def check_env() -> None:
    head("1. Interpreter and dependencies")

    v = sys.version_info
    if v < (3, 10):
        say(FAIL, f"Python {v.major}.{v.minor}", "3.10+ required (3.11 recommended)")
    else:
        say(OK, f"Python {v.major}.{v.minor}.{v.micro}")

    imports = [
        ("langchain_openai", "OpenAIEmbeddings"),
        ("langchain_openai", "ChatOpenAI"),
        ("langchain_community.chat_models", "ChatOllama"),
        ("langchain_community.embeddings", "OllamaEmbeddings"),
        ("langchain_community.document_loaders.csv_loader", "CSVLoader"),
        ("langchain_community.vectorstores", "FAISS"),
        ("langchain_classic.chains", "RetrievalQA"),
        ("langchain_classic.chains", "create_retrieval_chain"),
        ("langchain_classic.chains.combine_documents",
         "create_stuff_documents_chain"),
        ("langchain_classic.agents", "initialize_agent"),
        ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
        ("langchain_core.output_parsers", "StrOutputParser"),
        ("langchain_ollama", "ChatOllama"),
        ("faiss", None), ("pandas", None), ("yaml", None),
        ("dotenv", None), ("tiktoken", None), ("gurobipy", None),
    ]
    broken = []
    for mod, attr in imports:
        try:
            m = importlib.import_module(mod)
            if attr:
                getattr(m, attr)
        except Exception as e:                                    # noqa: BLE001
            broken.append(f"{mod}"
                          f"{'.' + attr if attr else ''}: "
                          f"{type(e).__name__}: {str(e)[:90]}")
    if broken:
        say(FAIL, f"{len(broken)} import(s) failed",
            "\n".join(broken) + "\n"
            "install with: pip install -r ../requirements-server.txt")
    else:
        say(OK, f"all {len(imports)} pipeline imports resolve")

    # Version drift is the failure this catches: the notebooks use the
    # LangChain 1.x package split, and a floated install silently lands on a
    # layout where `langchain_classic` means something else.
    req = PKG_ROOT / "requirements-server.txt"
    if not req.exists():
        say(WARN, "requirements-server.txt not found", "skipping version check")
        return
    from importlib.metadata import version, PackageNotFoundError
    drift = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, want = line.split("==", 1)
        try:
            got = version(name)
        except PackageNotFoundError:
            drift.append(f"{name}: not installed (want {want})")
            continue
        if got != want:
            drift.append(f"{name}: installed {got}, pinned {want}")
    if drift:
        say(WARN, f"{len(drift)} package version(s) differ from the pins",
            "\n".join(drift) + "\n"
            "usually harmless for patch versions; a different minor version of "
            "any langchain* package is not")
    else:
        say(OK, "installed versions match every pin")


# --------------------------------------------------------------------------- #
# 2. configuration
# --------------------------------------------------------------------------- #
def check_config():
    head("2. Configuration")
    sys.path.insert(0, str(HERE))
    try:
        import leanopt_exp as lx
    except Exception as e:                                        # noqa: BLE001
        say(FAIL, "cannot import leanopt_exp", f"{type(e).__name__}: {e}")
        return None

    cfg = None
    embeddings: dict[str, tuple] = {}
    for profile in ("gpt-oss-20b", "openrouter-gpt-oss-20b"):
        try:
            c = lx.load_config("exp_config.yaml", model_profile=profile,
                               dataset="Large-Scale-OR")
        except Exception as e:                                    # noqa: BLE001
            say(FAIL, f"profile '{profile}' does not load",
                f"{type(e).__name__}: {e}")
            continue
        spec = c.spec("coder")
        say(OK, f"profile '{profile}' loads",
            f"model      {spec.model}\n"
            f"endpoint   {spec.base_url}\n"
            f"max_tokens {spec.max_tokens}\n"
            f"extra      {spec.extra}\n"
            f"embeddings {c.embedding_provider}/{c.embedding_model}")
        embeddings[profile] = (c.embedding_provider, c.embedding_model)

        if profile == "gpt-oss-20b":
            cfg = c
            # These two are the whole reason this profile is not just
            # "the same config with a different model name".
            if spec.extra.get("num_ctx", 0) < 32768:
                say(FAIL, "num_ctx is missing or too small",
                    "prompts here average ~19k tokens; Ollama truncates "
                    "silently when the window is short")
            if spec.extra.get("reasoning") is not True:
                say(WARN, "reasoning is not True",
                    "chain-of-thought may land in `content` and collapse every "
                    "instance to the Others fallback")

    # The two gpt-oss routes must retrieve identically -- a fallback that
    # embeds differently from the route it replaces is a third arm, not a
    # fallback, and mixing runs from both would put two retrieval components
    # inside one column of the results table.
    if len(set(embeddings.values())) > 1:
        say(FAIL, "the two gpt-oss profiles use different embeddings",
            "\n".join(f"{k}: {v[0]}/{v[1]}" for k, v in embeddings.items()))
    elif embeddings:
        provider, model = next(iter(embeddings.values()))
        if (provider, model) == ("openai", "text-embedding-3-small"):
            say(OK, f"retrieval: {provider}/{model}",
                "same embedding model as the GPT-4.1 arm")
        else:
            # Deliberate as of 2026-08-20: local embeddings keep the whole arm
            # key-free. Restated on every run because it is a caveat the paper
            # has to carry, not a defect to fix.
            say(OK, f"retrieval: {provider}/{model} (local, no key needed)",
                "the GPT-4.1 arm embeds with openai/text-embedding-3-small, so "
                "the two arms retrieve from different vector spaces -- report "
                "this arm's retrieval component explicitly")

    if cfg is not None:
        say(OK, "agent limits",
            f"max_iterations {cfg.agent_max_iterations}, "
            f"max_execution_time {cfg.agent_max_execution_time}s, "
            f"sleep_between_instances {cfg.sleep_between_instances_s}s")
        note = str(cfg.hardware_note or "")
        if note.startswith("FILL IN"):
            say(WARN, "hardware_note not filled in yet",
                "fill it after the first run -- it is quoted in the paper")

    if IS_LOTO and cfg is not None:
        holdout = cfg.classification.get("holdout_type")
        if holdout in (None, "", "null"):
            say(OK, "holdout_type is null", "correct resting state between folds")
        else:
            say(FAIL, f"holdout_type is '{holdout}'",
                "a previous fold did not restore it; set it back to null "
                "before running anything")
    return cfg


# --------------------------------------------------------------------------- #
# 3. data integrity
# --------------------------------------------------------------------------- #
def check_data() -> None:
    head("3. Data")
    import pandas as pd

    if not DATASET.exists():
        say(FAIL, f"{DATASET} missing")
        return
    df = pd.read_csv(DATASET)
    digest = sha256_file(DATASET)[:16]
    if len(df) == DATASET_ROWS:
        say(OK, f"benchmark has {DATASET_ROWS} rows", f"sha256:{digest}")
    else:
        say(FAIL, f"benchmark has {len(df)} rows, expected {DATASET_ROWS}",
            "a truncated or substituted dataset produces a number that looks "
            "fine and answers a different question")

    manifest_path = PKG_ROOT / "package_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {}).get(TASK, {})
        bad, missing = [], []
        for rel, want in files.items():
            p = HERE / rel
            if not p.exists():
                missing.append(rel)
            elif sha256_file(p) != want:
                bad.append(rel)
        if missing or bad:
            say(FAIL, f"{len(missing)} file(s) missing, {len(bad)} modified",
                "\n".join([f"missing: {m}" for m in missing[:10]]
                          + [f"changed: {b}" for b in bad[:10]]))
        else:
            say(OK, f"all {len(files)} packaged files match the manifest")
    else:
        say(WARN, "package_manifest.json not found", "skipping integrity check")

    if IS_AIRNRM:
        # Air-NRM has no Dataset_address column; its inputs are fixed paths.
        gone = [p for p in AIRNRM_FIXED_PATHS if not Path(p).exists()]
        if gone:
            say(FAIL, f"{len(gone)} input file(s) the pipeline opens are missing",
                "\n".join(gone))
        else:
            say(OK, f"all {len(AIRNRM_FIXED_PATHS)} fixed input files present")
        # Standing guard: no cell that actually executes may name a
        # small-scale file. Small-scale has two fare families and 187-seat
        # aircraft; large-scale has four families and per-flight capacity.
        # Mixing them does not crash -- it silently answers a different
        # question, which is how this went unnoticed once already.
        import ast as _ast
        nb_cells = json.loads((HERE / NOTEBOOK).read_text(encoding="utf-8"))["cells"]
        leaked = []
        for idx, cell in enumerate(nb_cells):
            if cell.get("cell_type") != "code":
                continue
            body = "".join(cell.get("source", []))
            if not body.strip() or body.lstrip().startswith(("!", "%")):
                continue
            if "Batch_Process_Queries(" in body and "def " not in body.split(
                    "Batch_Process_Queries(")[0][-40:]:
                break          # the batch cell; run_notebook stops before it
            try:
                tree = _ast.parse(body)
            except SyntaxError:
                continue
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Constant) and isinstance(node.value, str) \
                        and "small_scale" in node.value:
                    leaked.append(f"cell {idx}: {node.value}")
        if leaked:
            say(FAIL, "the executed cells read small-scale data",
                "\n".join(leaked) + "\n"
                "this task must run entirely off Test_Dataset/Air_NRM/large_scale/")
        else:
            say(OK, "executed cells read large-scale data only",
                "no live small_scale path in any cell that runs")

        if "Label-objective" in df.columns and df["Label-objective"].notna().all():
            say(OK, "all 15 queries carry a ground-truth objective",
                "without these the run cannot be scored at all")
        else:
            say(FAIL, "queries are missing Label-objective values",
                "run link_air_nrm_labels.py in the source repo before shipping")
        return

    # Every path the pipeline will actually open, resolved ahead of time. A
    # missing CSV surfaces mid-run as one failed instance among a hundred,
    # which is easy to read as "the model got that one wrong".
    referenced, missing = set(), []
    for col in df.columns:
        if "address" in col.lower():
            for cell in df[col].dropna():
                referenced.update(x.strip() for x in str(cell).splitlines()
                                  if x.strip())
    for csv in sorted(Path("Large_Scale_Or_Files").glob("*.csv")):
        try:
            ref = pd.read_csv(csv)
        except Exception:                                         # noqa: BLE001
            continue
        for col in ref.columns:
            if "address" in col.lower():
                for cell in ref[col].dropna():
                    referenced.update(x.strip() for x in str(cell).splitlines()
                                      if x.strip())
    for path in sorted(referenced):
        if path in KNOWN_BAD_PATHS:
            continue
        if not Path(path).exists():
            missing.append(path)
    if missing:
        say(FAIL, f"{len(missing)} referenced data file(s) not found",
            "\n".join(missing[:15]))
    else:
        say(OK, f"all {len(referenced) - len(KNOWN_BAD_PATHS & referenced)} "
                f"referenced data files resolve")


# --------------------------------------------------------------------------- #
# 4. notebook
# --------------------------------------------------------------------------- #
def check_notebook() -> None:
    head("4. Notebook")
    nb_path = HERE / NOTEBOOK
    if not nb_path.exists():
        say(FAIL, f"{NOTEBOOK} missing")
        return
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    bad = []
    n = 0
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip() or src.lstrip().startswith(("!", "%")):
            continue
        n += 1
        try:
            ast.parse(src)
        except SyntaxError as e:
            bad.append(f"cell {i}: {e}")
    if bad:
        say(FAIL, f"{len(bad)} cell(s) do not parse", "\n".join(bad))
    else:
        say(OK, f"all {n} code cells parse")


def check_names() -> None:
    """
    Names used by the executed cells that nothing ever defines.

    `ast.parse` proves a cell is syntactically valid; it says nothing about
    whether the notebook can run from a clean kernel. On 2026-08-22 the
    gpt-oss Large-Scale-OR notebook called `split_by_newline(...)` while the
    only definition of it sat inside a commented-out block -- fine in the
    author's session, where the function was still in memory from an earlier
    execution, and an instant NameError anywhere else. All 22 LOTO seeds died
    12 seconds in, on a machine on the other side of the world.

    The scan is deliberately conservative: a name counts as defined if it is
    bound anywhere in the executed sequence, at any nesting level. That misses
    order-of-definition problems but produces no false positives, which is what
    makes it worth having a machine run it.
    """
    head("5. Clean-kernel name resolution")
    import builtins

    nb = json.loads((HERE / NOTEBOOK).read_text(encoding="utf-8"))
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        body = "".join(cell.get("source", []))
        if not body.strip() or body.lstrip().startswith(("!", "%")):
            continue
        if BATCH_CALL_RE.search(body):
            break                     # the drivers stop before this cell
        cells.append((i, body))

    bound, used = set(), {}
    for idx, body in cells:
        try:
            tree = ast.parse(body)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    bound.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used.setdefault(node.id, idx)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                bound.add(node.name)
                for a in list(getattr(node, "args", ast.arguments(
                        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[],
                        defaults=[])).args or []):
                    bound.add(a.arg)
                args = getattr(node, "args", None)
                if args is not None:
                    for a in (args.posonlyargs + args.args + args.kwonlyargs):
                        bound.add(a.arg)
                    for a in (args.vararg, args.kwarg):
                        if a:
                            bound.add(a.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.Global):
                bound.update(node.names)

    # Module dunders exist in any namespace but are not builtins attributes.
    ALWAYS = {"__builtins__", "__name__", "__file__", "__doc__", "__spec__"}
    missing = {n: i for n, i in used.items()
               if n not in bound and n not in ALWAYS and not hasattr(builtins, n)}
    if missing:
        say(FAIL, f"{len(missing)} name(s) used but never defined",
            "\n".join(f"{n}  (first used in cell {i})"
                       for n, i in sorted(missing.items(), key=lambda kv: kv[1]))
            + "\nthe notebook cannot run from a clean kernel")
    else:
        say(OK, f"every name the {len(cells)} executed cells use is defined")


# --------------------------------------------------------------------------- #
# 6. LOTO leakage sealing (offline, no model)
# --------------------------------------------------------------------------- #
def check_loto_sealing() -> None:
    head("5. LOTO leakage sealing")
    import leanopt_exp as lx

    nb = json.loads((HERE / NOTEBOOK).read_text(encoding="utf-8"))
    cell6 = "".join(nb["cells"][6]["source"])
    try:
        region = cell6[cell6.index("ALLOWED_CATS = ["):
                       cell6.index("def classify_problem(")]
    except ValueError:
        say(FAIL, "cannot locate the classifier prompt region in cell 6",
            "the notebook was edited; the sealing check cannot run")
        return

    folds = [None, "Network Revenue Management", "Resource Allocation",
             "Facility Location Problem", "Transportation Problem",
             "Assignment Problem", "Mixture"]
    baseline, problems, sealed = None, [], []

    for holdout in folds:
        cfg = lx.load_config("exp_config.yaml", model_profile="gpt-oss-20b",
                             dataset="Large-Scale-OR")
        cfg.classification["holdout_type"] = holdout
        ns = {"lx": lx, "CFG": cfg, "re": re}
        try:
            # The cell prints its own [LOTO] chatter; silence it, the summary
            # below is what matters.
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                exec(region, ns)                                  # noqa: S102
        except AssertionError as e:
            problems.append(f"{holdout}: own assertion fired -- {e}")
            continue
        except Exception as e:                                    # noqa: BLE001
            problems.append(f"{holdout}: {type(e).__name__}: {e}")
            continue

        msg = ns["CLASSIFY_SYS_MSG"]
        if holdout is None:
            baseline = msg
            continue
        low = msg.lower()
        for alias in lx.holdout_alias_group(cfg):
            if re.search(rf"\b{re.escape(alias)}\b", low):
                problems.append(f"{holdout}: '{alias}' still appears in the "
                                f"classifier prompt")
        # Every fold but RA must leave the prompt text otherwise untouched --
        # RA is the one whose rule 3 names a type and has to be reworded.
        if holdout != "Resource Allocation" and baseline is not None:
            base_rules = [l for l in baseline.splitlines() if l.startswith("2.")]
            now_rules = [l for l in msg.splitlines() if l.startswith("2.")]
            if base_rules != now_rules:
                problems.append(f"{holdout}: prompt text drifted from the "
                                f"no-holdout baseline")
        sealed.append(str(holdout))

    if problems:
        say(FAIL, f"{len(problems)} sealing problem(s)", "\n".join(problems))
    else:
        say(OK, f"all {len(sealed)} folds sealed", ", ".join(sealed))


# --------------------------------------------------------------------------- #
# 6. output directory and secrets
# --------------------------------------------------------------------------- #
def check_runs_dir() -> None:
    head("6. Output directory")
    runs = HERE / "runs"
    runs.mkdir(exist_ok=True)
    probe = runs / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        say(OK, "runs/ writable")
    except Exception as e:                                        # noqa: BLE001
        say(FAIL, "runs/ not writable", f"{type(e).__name__}: {e}")

    existing = [p.name for p in runs.iterdir() if p.is_dir()]
    if existing:
        say(WARN, f"runs/ already holds {len(existing)} run(s)",
            "fine when resuming; unexpected in a freshly unpacked package")


SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")


def check_no_secrets() -> None:
    head("7. Secrets")
    hits, envs = [], []
    for path in PKG_ROOT.rglob("*"):
        if not path.is_file() or "runs" in path.parts:
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            envs.append(str(path.relative_to(PKG_ROOT)))
            continue
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".json", ".md",
                                       ".sh", ".txt", ".ipynb", ".example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:                                         # noqa: BLE001
            continue
        if SECRET_RE.search(text):
            hits.append(str(path.relative_to(PKG_ROOT)))
    if envs:
        say(FAIL, f"{len(envs)} .env file(s) inside the package",
            "\n".join(envs) + "\ndelete before sending this anywhere")
    if hits:
        say(FAIL, f"possible API key(s) in {len(hits)} file(s)",
            "\n".join(hits))
    if not envs and not hits:
        say(OK, "no .env files and no key-shaped strings in the package")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 62)
    print(f"  package self-check -- task: {TASK}")
    print(f"  directory: {HERE}")
    print("=" * 62)

    check_env()
    cfg = check_config()
    check_data()
    check_notebook()
    check_names()
    if IS_LOTO and cfg is not None:
        check_loto_sealing()
    check_runs_dir()
    check_no_secrets()

    print("\n" + "=" * 62)
    print(f"  {_counts[OK]} OK   {_counts[WARN]} WARN   {_counts[FAIL]} FAIL")
    if _counts[FAIL]:
        print("\n  Fix the FAILs before running anything.")
    else:
        print(f"\n  Package looks sound. Next, from the package root:")
        print(f"      bash setup_and_smoke.sh {TASK}")
        print("  which pulls the models if needed, runs a couple of instances")
        print("  and grades them. Do not start the full runs before that.")
    print("=" * 62)
    return _counts[FAIL]


if __name__ == "__main__":
    sys.exit(main())
