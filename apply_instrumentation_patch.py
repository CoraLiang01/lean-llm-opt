#!/usr/bin/env python3
"""
apply_instrumentation_patch.py
==============================
Rewrites the experiment notebooks to use the unified configuration
(`exp_config.yaml`) and the run logger (`leanopt_exp.py`).

    python apply_instrumentation_patch.py --check     # dry run, report only
    python apply_instrumentation_patch.py             # rewrite the notebooks

Reads the pristine notebooks from `original_notebooks/` and writes the
instrumented versions to the repository root, under the same file names. Cell
outputs are cleared in the copy, because the stored outputs came from the
pre-patch code and would be misleading.

!! This OVERWRITES the notebooks in the repository root. They are generated
   files: `original_notebooks/` + this script is the source of truth. If you
   edit a root notebook by hand, either port the change into a rule here or
   do not run this script afterwards. `--check` never writes anything.

What the patch does
-------------------
1. Replaces the hard-coded `user_api_key` with a config block that builds
   CFG / LOG / EMBEDDINGS.
2. Routes every ChatOpenAI / ChatOllama construction through
   `lx.build_llm(CFG, role)` -- role inferred from the variable name
   (llm1 -> classifier, llm2 -> modeler, llm -> data_agent, llm_code -> coder).
   This is what unifies the classifier model: the notebooks used gpt-4 in the
   main pipeline and gpt-4.1 in the ablations.
3. Routes every embedding construction through `EMBEDDINGS`.
4. Routes literal-k `as_retriever` calls through `lx.build_retriever`, keeping
   each call site's original k (recorded in exp_config.yaml:retrievers).
5. Adds max_iterations / max_execution_time / early_stopping to every agent.
6. Rewrites the batch loops so each instance is wrapped in
   `LOG.instance(...)` with `lx.stage(...)` markers, and so a failed instance
   still appends a row (the old `except: continue` dropped rows, which made
   the result columns line up with the wrong queries).
7. Redirects result CSVs into the run directory.

Every rule declares how many matches it expects; a mismatch aborts the patch
rather than silently producing a half-instrumented notebook.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Pristine (pre-instrumentation) notebooks live here. They were moved out of the
# repo root during the 2026-07 tidy-up: the root now holds only the instrumented
# notebooks coauthors are meant to run, under the original file names. The
# fallback keeps this script working on older checkouts where the originals
# still sit next to it.
ORIG_DIR = HERE / "original_notebooks"


def _orig(name: str) -> Path:
    """Resolve the path of an original, un-instrumented notebook."""
    p = ORIG_DIR / name
    if p.exists():
        return p
    legacy = HERE / name
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"original notebook not found: looked in {ORIG_DIR} and {HERE} for {name}"
    )

# --------------------------------------------------------------------------- #
# shared source fragments
# --------------------------------------------------------------------------- #

CONFIG_BLOCK = '''
# ---------------------------------------------------------------------------
# Unified experiment configuration + run logging  (see PATCH_GUIDE.md)
# ---------------------------------------------------------------------------
import os
import leanopt_exp as lx

# API keys come from a .env file at the repository root (see .env.example),
# from the shell environment, or -- if neither is set -- from a hidden prompt.
# Nothing is hard-coded here and nothing is written to the run log.
try:
    from dotenv import load_dotenv
    # override=True: the .env file wins over anything already in the
    # environment. Without it, a stale key left in os.environ by an earlier
    # run silently shadows the value you just edited into .env.
    load_dotenv(override=True)
except ImportError:
    pass

CFG = lx.load_config(
    "exp_config.yaml",
    model_profile={profile!r},
    method={method!r},
    dataset={dataset!r},
    repeat_index=0,
)
lx.ensure_api_keys(CFG)                          # only for providers this profile uses
user_api_key = os.environ.get("OPENAI_API_KEY", "")   # legacy references

LOG = lx.RunLogger(CFG)
EMBEDDINGS = lx.build_embeddings(CFG)
'''

ROLE_BY_VAR = {
    "llm1": "classifier",
    "llm2": "modeler",
    "llm": "data_agent",
    "llm_code": "coder",
    "llm_classify": "classifier",
    "llm_codegen": "coder",
}

APIKEY_RE = re.compile(
    r'^user_api_key\s*=\s*["\'].*?["\'].*$', re.M)

CHATMODEL_RE = re.compile(
    r'(?P<var>\w+)\s*=\s*(?:ChatOpenAI|ChatOllama)\s*\((?:[^()]|\([^()]*\))*\)',
    re.S)

EMB_RE = re.compile(
    r'(?:OpenAIEmbeddings|OllamaEmbeddings)\s*\((?:[^()]|\([^()]*\))*\)', re.S)

# literal-k retriever calls only; parameterised ones ({"k": k}) are left alone
RETRIEVER_RE = re.compile(
    r'(?P<store>[\w.]+)\.as_retriever\(\s*'
    r'(?P<args>(?:[^()]|\([^()]*\)|\{[^{}]*\}|\{[^{}]*\{[^{}]*\}[^{}]*\})*?)\s*\)',
    re.S)

AGENT_TAIL_RE = re.compile(
    r'agent_kwargs\s*=\s*\{(?P<ak>[^{}]*)\}\s*,\s*'
    r'verbose\s*=\s*\w+\s*,\s*'
    r'handle_parsing_errors\s*=\s*\w+\s*,?\s*(?:#[^\n]*)?\s*'
    r'(?:max_iterations\s*=\s*\d+\s*,?\s*(?:#[^\n]*)?\s*)?\)', re.S)

TO_CSV_RE = re.compile(r'\.to_csv\(\s*(?P<q>["\'])(?P<name>[^"\']+\.csv)(?P=q)')

# Air-NRM data was split into two cases under one parent folder:
#   Test_Dataset/Air_NRM/small_scale/   the original 3-airport toy instance
#   Test_Dataset/Air_NRM/large_scale/   the SQ direct 2-city real-data build
# The originals in original_notebooks/ still point at the old flat layout, so
# the redirect happens here rather than by editing them.
AIR_NRM_PATH_RE = re.compile(
    r'Test_Dataset/Air_NRM/(?!small_scale/|large_scale/)')


# --------------------------------------------------------------------------- #
class Patcher:
    def __init__(self, path: Path, profile: str, method: str, dataset: str):
        self.path = path
        self.nb = json.loads(path.read_text(encoding="utf-8"))
        self.profile, self.method, self.dataset = profile, method, dataset
        self.report: list[str] = []

    # -- helpers --------------------------------------------------------- #
    def src(self, i: int) -> str:
        return "".join(self.nb["cells"][i]["source"])

    def set_src(self, i: int, s: str):
        self.nb["cells"][i]["source"] = s.splitlines(keepends=True)

    def note(self, msg: str):
        self.report.append(msg)

    def expect(self, got: int, want, label: str):
        ok = (got == want) if isinstance(want, int) else (got in want)
        if not ok:
            raise AssertionError(
                f"{self.path.name}: rule '{label}' matched {got} times, "
                f"expected {want}")
        self.note(f"  {label}: {got}")

    # -- rules ----------------------------------------------------------- #
    def insert_config(self, cell: int):
        s = self.src(cell)
        if not APIKEY_RE.search(s):
            raise AssertionError(f"{self.path.name}: no user_api_key line in "
                                 f"cell {cell}")
        block = CONFIG_BLOCK.format(profile=self.profile, method=self.method,
                                    dataset=self.dataset)
        s = APIKEY_RE.sub(block.strip(), s, count=1)
        self.set_src(cell, s)
        self.note(f"  config block -> cell {cell}")

    def append_config(self, cell: int):
        """For notebooks with no user_api_key line."""
        block = CONFIG_BLOCK.format(profile=self.profile, method=self.method,
                                    dataset=self.dataset)
        self.set_src(cell, self.src(cell).rstrip() + "\n\n" + block.strip() + "\n")
        self.note(f"  config block appended to cell {cell}")

    def patch_air_nrm_paths(self, want):
        """Redirect Air-NRM reads into Test_Dataset/Air_NRM/small_scale/.

        These notebooks all run the small (3-airport) case. The large_scale/
        sibling holds the SQ real-data build and is not read by any notebook
        yet -- it is driven by its own build scripts.
        """
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s, k = AIR_NRM_PATH_RE.subn(
                "Test_Dataset/Air_NRM/small_scale/", self.src(i))
            if k:
                self.set_src(i, s)
                n += k
        self.expect(n, want, "Air-NRM -> small_scale paths")

    def patch_models(self, cells=None, want=None):
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code" or (cells and i not in cells):
                continue
            s = self.src(i)

            def repl(m):
                nonlocal n
                var = m.group("var")
                role = ROLE_BY_VAR.get(var)
                if role is None:
                    return m.group(0)
                n += 1
                return f'{var} = lx.build_llm(CFG, "{role}")'

            s2 = CHATMODEL_RE.sub(repl, s)
            if s2 != s:
                self.set_src(i, s2)
        self.expect(n, want if want is not None else n, "build_llm")

    def patch_embeddings(self, want=None):
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            s2, k = EMB_RE.subn("EMBEDDINGS", s)
            if k:
                n += k
                self.set_src(i, s2)
        self.expect(n, want if want is not None else n, "embeddings")

    def patch_retrievers(self, mapping: dict):
        """mapping: {cell_index: [key, key, ...]} in order of appearance."""
        total = 0
        for cell, keys in mapping.items():
            s = self.src(cell)
            it = iter(keys)
            used = []

            def repl(m):
                args = m.group("args")
                if re.search(r'["\']k["\']\s*:\s*\d+', args) is None:
                    return m.group(0)          # parameterised k -> leave alone
                try:
                    key = next(it)
                except StopIteration:
                    raise AssertionError(
                        f"{self.path.name} cell {cell}: more literal-k "
                        f"retrievers than keys supplied")
                used.append(key)
                filt = re.search(r'["\']filter["\']\s*:\s*(\{.*?\})\s*\}', args,
                                 re.S)
                store = m.group("store")
                if filt:
                    return (f'lx.build_retriever(CFG, {store}, "{key}", '
                            f'search_kwargs={{"filter": {filt.group(1)}}})')
                return f'lx.build_retriever(CFG, {store}, "{key}")'

            s2 = RETRIEVER_RE.sub(repl, s)
            leftover = list(it)
            if leftover:
                raise AssertionError(
                    f"{self.path.name} cell {cell}: keys unused {leftover}")
            self.set_src(cell, s2)
            total += len(used)
            self.note(f"  retrievers cell {cell}: {used}")
        self.note(f"  retrievers total: {total}")

    def patch_agents(self, want=None):
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)

            def repl(m):
                nonlocal n
                ak = m.group("ak")
                # Use the ACTUAL prefix/suffix expressions from the call site.
                # They are not always named `prefix`/`suffix` -- the few-shot
                # ablation uses PREFIX_CA / PREFIX_NP / SUFFIX.
                pre = re.search(r'["\']prefix["\']\s*:\s*([^,\n}]+)', ak)
                suf = re.search(r'["\']suffix["\']\s*:\s*([^,\n}]+)', ak)
                if not (pre and suf):
                    return m.group(0)
                n += 1
                iv = re.search(r'["\']input_variables["\']\s*:\s*(\[[^\]]*\])',
                               ak)
                extra = f", input_variables={iv.group(1)}" if iv else ""
                return (f'**lx.agent_kwargs(CFG, {pre.group(1).strip()}, '
                        f'{suf.group(1).strip()}{extra}),\n    )')

            s2 = AGENT_TAIL_RE.sub(repl, s)
            if s2 != s:
                self.set_src(i, s2)
        self.expect(n, want if want is not None else n, "agent_kwargs")

    def patch_read_back(self, want=None):
        """
        Results are now written into runs/<run_id>/, so the later
        read_and_combine_csvs(file_order) calls -- which pass bare file names --
        must resolve them against LOG.dir too. Without this the write path and
        the read path disagree and the combine step silently finds no files.
        """
        old = "    for fname in file_order:\n        if os.path.exists(fname):"
        new = ("    for fname in file_order:\n"
               "        if not os.path.exists(fname):\n"
               "            fname = LOG.dir / fname      # results live in the run dir\n"
               "        if os.path.exists(fname):")
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            if old in s:
                self.set_src(i, s.replace(old, new))
                n += 1
        self.expect(n, want if want is not None else n, "read_and_combine -> LOG.dir")

    def patch_classification_docs(self, want=None):
        """
        The classification step retrieved over documents built by CSVLoader,
        i.e. every RefData column including `Label` (the full mathematical
        model) and `Label_Code`. With k=5 and chain_type="stuff" that pastes
        five complete formulations into a prompt whose only job is to name the
        problem class -- ~2300 prompt tokens per call, of which ~970 carry no
        information about the class.

        Replaced by lx.load_refdata_docs(CFG), which keeps only the columns
        listed in exp_config.yaml:classification.refdata_columns.
        """
        import re as _re
        pats = [
            # 5 notebooks: loader1 = CSVLoader(...RefData...); refdata = loader1.load()
            (_re.compile(
                r'[ \t]*loader1\s*=\s*CSVLoader\(\s*file_path\s*=\s*["\'][^"\']*RefData[^"\']*["\']'
                r'[^)]*\)\s*\n[ \t]*refdata\s*=\s*loader1\.load\(\)'),
             '    # classification only needs the problem description + its class label\n'
             '    refdata = lx.load_refdata_docs(CFG)'),
            # 4.1 large-scale: loader = CSVLoader(...RefData...); data = loader.load()
            (_re.compile(
                r'[ \t]*loader\s*=\s*CSVLoader\(\s*file_path\s*=\s*["\'][^"\']*RefData[^"\']*["\']'
                r'[^)]*\)\s*\n[ \t]*data\s*=\s*loader\.load\(\)'),
             '# classification only needs the problem description + its class label\n'
             'data = lx.load_refdata_docs(CFG)'),
            # oss large-scale: ref_docs = CSVLoader(REF_CSV_PATH...).load()
            (_re.compile(
                r'[ \t]*ref_docs\s*=\s*CSVLoader\([^)]*\)\.load\(\)'),
             '# full rows (Label + Data_address) stay available for few-shot building\n'
             'ref_docs = CSVLoader(file_path=REF_CSV_PATH, encoding="utf-8").load()\n'
             '# slim view used for CLASSIFICATION only\n'
             'cls_docs = lx.load_refdata_docs(CFG)'),
        ]
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            for pat, rep in pats:
                s2, k = pat.subn(rep, s)
                if k:
                    n += k
                    s = s2
            if s != self.src(i):
                self.set_src(i, s)
        self.expect(n, want if want is not None else n, "classification docs")

    def patch_oss_classification(self, want=None):
        """
        The gpt-oss notebook classifies with build_dynamic_few_shot(), which
        pastes the retrieved rows' `Label` (full formulation) AND reads every
        file listed in their `Data_address` into the prompt. For deciding a
        problem class that is all dead weight.

        Adds a slim classification-only few-shot builder over the reduced
        RefData view and points classify_problem at it. build_dynamic_few_shot
        is left untouched so the modelling stages keep their behaviour.
        """
        import re as _re
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            if "cls_docs = lx.load_refdata_docs(CFG)" not in s:
                continue

            helper = (
                'cls_docs = lx.load_refdata_docs(CFG)\n'
                'cls_store: FAISS = FAISS.from_documents(cls_docs, embeddings)\n'
                '\n'
                '\n'
                'def build_classification_few_shot(query: str, k: int = None) -> str:\n'
                '    """\n'
                '    Few-shot block for CLASSIFICATION only: problem description ->\n'
                '    class label. No formulations and no data files, because neither\n'
                '    tells you the problem class.\n'
                '    """\n'
                '    retriever = lx.build_retriever(CFG, cls_store, "oss_refdata")\n'
                '    return "\\n\\n".join(d.page_content for d in retriever.invoke(query))\n'
            )
            s2 = s.replace('cls_docs = lx.load_refdata_docs(CFG)', helper, 1)

            s2, k = _re.subn(
                r'few_shot_dynamic\s*=\s*build_dynamic_few_shot\(\s*user_query\s*,\s*k\s*=\s*\d+\s*\)',
                'few_shot_dynamic = build_classification_few_shot(user_query)',
                s2)
            n += k
            self.set_src(i, s2)
        self.expect(n, want if want is not None else n, "oss classification few-shot")

    def append_close_cell(self):
        """
        Nothing in the original notebooks called LOG.close(), so summary.json
        was never written. instances.jsonl / calls.jsonl are flushed per
        instance and are what aggregate_runs.py reads, so no data was at risk,
        but the run-level total was missing. close() is idempotent, so this
        cell can be run after every slice of a batch.
        """
        src = (
            "# ---------------------------------------------------------------\n"
            "# Finish the run: writes runs/<run_id>/summary.json and prints the\n"
            "# totals. Safe to run after every slice.\n"
            "# ---------------------------------------------------------------\n"
            "agg = LOG.close()\n"
            "\n"
            "import json\n"
            "rows = [json.loads(l) for l in open(LOG.dir_path / 'instances.jsonl')]\n"
            "est = sum(r['summary'].get('n_estimated_token_calls', 0) for r in rows)\n"
            "print(f\"\\ntoken source: {'provider (good)' if est == 0 else f'{est} calls ESTIMATED -- report as estimates'}\")\n"
            "print('run dir:', LOG.dir_path)\n"
        )
        self.nb["cells"].append({
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True),
        })
        self.note("  appended LOG.close() cell")

    def patch_gurobi_import(self, want=None):
        """
        run_gurobi_code() builds its exec environment from the notebook globals
        `gp` and `GRB`, but nothing in the Large-Scale-OR notebook ever imports
        gurobipy -- the `import gurobipy as gp` lines that grep finds are text
        inside the few-shot prompt strings. On a fresh kernel the very first
        call raised NameError: name 'gp' is not defined, which the bare
        `except Exception` turned into "Execution error", so every objective
        value came back None.
        """
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            if "def run_gurobi_code" not in s:
                continue
            lines = [ln for ln in s.split("\n")
                     if ln.strip().startswith(("import gurobipy",
                                               "from gurobipy"))
                     and not ln.startswith(" ")]
            if lines:
                continue                      # already imported at top level
            header = ("# gurobipy must be imported here: run_gurobi_code() "
                      "injects gp / GRB\n"
                      "# into the exec environment of the generated code.\n"
                      "import gurobipy as gp\n"
                      "from gurobipy import GRB\n\n")
            self.set_src(i, header + s)
            n += 1
        self.expect(n, want if want is not None else n, "gurobipy import")

    def patch_to_csv(self, want=None):
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            out_lines, k = [], 0
            for line in s.split("\n"):
                if line.lstrip().startswith("#"):
                    out_lines.append(line)
                    continue
                line2, kk = TO_CSV_RE.subn(
                    lambda m: f'.to_csv(LOG.dir / "{m.group("name")}"', line)
                k += kk
                out_lines.append(line2)
            s2 = "\n".join(out_lines)
            if k:
                n += k
                self.set_src(i, s2)
        self.expect(n, want if want is not None else n, "to_csv -> LOG.dir")

    # -- Air-NRM specific surgery ---------------------------------------- #
    def patch_process_input(self, cell: int):
        """
        Surgical edit of Process_Input, rather than a blanket replacement:
        each notebook calls its own downstream functions (ProcessCA vs.
        Fewshot_Only_CA), so we only add the stage markers and the missing
        else branch.
        """
        s = self.src(cell)
        if "def Process_Input" not in s:
            raise AssertionError(f"cell {cell} has no Process_Input")

        # 1. bind Type / output_model up front (was UnboundLocalError when the
        #    category did not match "Sales-Based Linear Programming")
        s = re.sub(r'(def Process_Input\([^)]*\):\n)',
                   r'\1  Type, output_model = None, None\n', s, count=1)

        # 2. stage-tag the classification call
        s = re.sub(r'^(\s*)category_original\s*=\s*(Problemtype\(query\))\s*$',
                   lambda m: (f'{m.group(1)}with lx.stage("classification"):\n'
                              f'{m.group(1)}    category_original = {m.group(2)}'),
                   s, count=1, flags=re.M)

        # 3. stage-tag the modelling calls
        def wrap_model(m):
            ind, call = m.group(1), m.group(2)
            return (f'{ind}with lx.stage("modeling"):\n'
                    f'{ind}    output_model = {call}')

        s, n = re.subn(r'^(\s*)output_model\s*=\s*(\w+\(query\))\s*$',
                       wrap_model, s, flags=re.M)
        if n == 0:
            raise AssertionError(f"cell {cell}: no output_model assignment found")

        # 4. fail loudly instead of raising UnboundLocalError
        s = re.sub(r'^(\s*)return Type\s*,\s*output_model\s*$',
                   lambda m: (
                       f'{m.group(1)}if output_model is None:\n'
                       f'{m.group(1)}    raise ValueError(\n'
                       f'{m.group(1)}        f"unroutable problem type: '
                       f'{{category_original!r}}")\n'
                       f'{m.group(1)}return Type, output_model'),
                   s, count=1, flags=re.M)
        self.set_src(cell, s)
        self.note(f"  Process_Input instrumented (cell {cell}, "
                  f"{n} modelling call(s))")

    def patch_batch_loop(self, cell: int):
        """Replace only the Batch_Process_Queries part of the cell."""
        s = self.src(cell)
        anchor = None
        for cand in ("from tqdm import tqdm", "def Batch_Process_Queries"):
            if cand in s:
                anchor = min(x for x in [s.find(cand)] if x >= 0)
                break
        if anchor is None:
            raise AssertionError(f"cell {cell} has no Batch_Process_Queries")
        self.set_src(cell, s[:anchor].rstrip() + "\n\n" + BATCH_AIR.lstrip("\n"))
        self.note(f"  Batch_Process_Queries instrumented (cell {cell})")

    def replace_cell(self, cell: int, new_src: str, must_contain: str = ""):
        if must_contain and must_contain not in self.src(cell):
            raise AssertionError(
                f"{self.path.name}: cell {cell} does not contain "
                f"{must_contain!r}; notebook layout changed?")
        self.set_src(cell, new_src.lstrip("\n"))
        self.note(f"  rewrote cell {cell}")

    def sub_all(self, pattern: str, repl: str, want, label: str, flags=0):
        n = 0
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            s2, k = re.subn(pattern, repl, s, flags=flags)
            if k:
                n += k
                self.set_src(i, s2)
        self.expect(n, want, label)

    # -- output ---------------------------------------------------------- #
    def validate(self):
        bad = []
        for i, c in enumerate(self.nb["cells"]):
            if c["cell_type"] != "code":
                continue
            s = self.src(i)
            if s.strip().startswith("!") or s.strip().startswith("%"):
                continue
            try:
                ast.parse(s)
            except SyntaxError as e:
                bad.append((i, e))
        if bad:
            for i, e in bad:
                print(f"  !! cell {i} syntax error: {e}", file=sys.stderr)
            raise AssertionError(f"{self.path.name}: {len(bad)} cells fail to "
                                 f"parse")
        self.note("  ast.parse: all code cells OK")

    def write(self, check_only: bool) -> Path:
        for c in self.nb["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []
                c["execution_count"] = None
        out = HERE / self.path.name
        if not check_only:
            out.write_text(json.dumps(self.nb, indent=1, ensure_ascii=False),
                           encoding="utf-8")
        return out


# --------------------------------------------------------------------------- #
# hand-written replacement cells
# --------------------------------------------------------------------------- #

RUN_TEST_LARGE = '''
def run_test(test, agent, log=LOG, cfg=CFG):
    """
    Instrumented batch loop.

    Changes vs. the original:
      * every instance is wrapped in log.instance(...) so tokens / calls /
        cost / latency are recorded per instance;
      * lx.stage(...) attributes each LLM call to a pipeline stage;
      * a failed instance still appends None to every list. The original
        `except requests.exceptions.RequestException: continue` skipped the
        append, so output_model/output_code/classification silently became
        shorter than `test`, and the later
        pd.DataFrame({'Query': test['Query'], 'model_output': output_model})
        lined the outputs up with the wrong queries;
      * the hard-coded time.sleep(15) is now cfg.sleep_between_instances_s.
    """
    output_model, output_code, classification = [], [], []

    def extract_problem_type(output_text):
        pattern = (r'(Network Revenue Management|Network Revenue Management Problem|'
                   r'Resource Allocation|Resource Allocation Problem|Transportation|'
                   r'Transportation Problem|Facility Location Problem|Assignment Problem|'
                   r'AP|Uncapacited Facility Location Problem|NRM|RA|TP|FLP|UFLP|'
                   r'Others without CSV|Sales-Based Linear Programming|SBLP|'
                   r'Others with CSV)')
        match = re.search(pattern, output_text, re.IGNORECASE)
        return match.group(0) if match else None

    def csv_detect(row):
        return 1 if 'Dataset_address' in row.index else 0

    for index, row in test.iterrows():
        query = row['Query']
        # 'Problem Type' is what Large-scale-or-101.csv actually calls it;
        # 'Type' is kept as a fallback for other test files.
        with log.instance(instance_id=int(index),
                          query=query,
                          gold_type=row.get('Type') or row.get('Problem Type'),
                          dataset_address=row.get('Dataset_address'),
                          size_class=row.get('Type by size'),
                          gold_objective=row.get('Label-objective')) as rec:
            output, code_response, selected_problem = None, None, None
            try:
                # ---------------- stage 1: problem classification ----------
                with lx.stage("classification"):
                    response = lx.call_with_retry(
                        agent.invoke, cfg,
                        f"What is the problem type of the text? text:{query}")
                selected_problem = extract_problem_type(response['output'])
                rec.set(pred_type=selected_problem)

                # ---------------- stage 2: type-tailored workflow ----------
                with lx.stage("modeling"):
                    if csv_detect(row):
                        dataset_address = row['Dataset_address']
                        if selected_problem in ("Network Revenue Management", "NRM",
                                                "Network Revenue Management Problem"):
                            print("----------Network Revenue Management-----------")
                            output = get_NRM_response(query, dataset_address)
                        elif selected_problem in ("Resource Allocation", "RA",
                                                  "Resource Allocation Problem"):
                            print("----------Resource Allocation-----------")
                            output = get_RA_response(query, dataset_address)
                        elif selected_problem in ("Transportation", "TP",
                                                  "Transportation Problem"):
                            print("----------Transportation-----------")
                            output = get_TP_response(query, dataset_address)
                        elif selected_problem in ("Facility Location Problem", "FLP",
                                                  "Uncapacited Facility Location",
                                                  "UFLP"):
                            print("----------Facility Location Problem-----------")
                            output = get_FLP_response(query, dataset_address)
                        elif selected_problem in ("Assignment Problem", "AP"):
                            print("----------Assignment Problem-----------")
                            output = get_AP_response(query, dataset_address)
                        else:
                            print("----------Others with CSV-----------")
                            output = get_Others_response(query, dataset_address)
                            selected_problem = 'Others with CSV'
                    else:
                        print("----------Others without CSV-----------")
                        output = get_others_without_CSV_response(query)

                # ---------------- stage 3: code generation -----------------
                with lx.stage("codegen"):
                    if output.strip().startswith("```python"):
                        print("[INFO] Output is already Gurobi code. Skipping get_code.")
                        code_response = output.strip().replace(
                            "```python", "").replace("```", "")
                    else:
                        code_response = get_code(output, selected_problem)

                rec.set(model_output=output, code_output=code_response)

            except Exception as e:
                print(f"[{index}] failed: {type(e).__name__}: {e}")
                rec.fail(e)                      # status -> "failed" in the log
                rec.set(model_output=output, code_output=code_response)

            # one row per instance, always -- keeps columns aligned
            output_model.append(output)
            output_code.append(code_response)
            classification.append(selected_problem)

    return output_model, output_code, classification
'''

RUN_TEST_OSS_LARGE = RUN_TEST_LARGE.replace(
    "def run_test(test, agent, log=LOG, cfg=CFG):",
    "def run_test(test, classify_problem, log=LOG, cfg=CFG):"
).replace(
    """                with lx.stage("classification"):
                    response = lx.call_with_retry(
                        agent.invoke, cfg,
                        f"What is the problem type of the text? text:{query}")
                selected_problem = extract_problem_type(response['output'])""",
    """                with lx.stage("classification"):
                    response = lx.call_with_retry(classify_problem, cfg, query)
                selected_problem = extract_problem_type(response)"""
)

BATCH_AIR = '''
import re as _re
from tqdm import tqdm


def _extract_gurobipy_code(text):
    """Pull the gurobipy block out of a model answer.

    Air-NRM answers carry the formulation and the code in one string. The
    original notebooks only separated them further down, in the cell that runs
    the whole test set -- so anything driving the batch loop directly (a slice
    for a smoke test, run_notebook.py) logged no code at all, and score_runs.py
    then reported every instance as unsolvable. Doing it here means the record
    is complete however the loop was started.
    """
    if not text:
        return None
    blocks = _re.findall(r"```(?:python)?(.*?)```", str(text), _re.DOTALL)
    for b in blocks:
        if "gurobipy" in b or "gurobi" in b.lower():
            return b.strip()
    return blocks[0].strip() if blocks else None


def Batch_Process_Queries(df, query_column='Query', log=LOG, cfg=CFG,
                          gold_column='Label-objective'):
    """
    Instrumented batch loop for the Air-NRM notebooks.

    Every query is wrapped in log.instance(...), so tokens / LLM calls /
    tool calls / cost / latency are recorded per instance and land in
    runs/<run_id>/instances.jsonl. Failures still produce a row, so the
    output frame always has the same length as the input frame.

    Two things beyond the original loop, both so the log is self-sufficient:
      * the ground-truth objective is carried through when the query file has
        one (see link_air_nrm_labels.py), otherwise score_runs.py has nothing
        to compare against;
      * the gurobipy block is extracted per instance rather than only in the
        run-everything cell downstream.
    """
    results = []
    has_gold = gold_column in df.columns

    for pos, (idx, row) in enumerate(tqdm(list(df.iterrows()),
                                          desc="Processing Queries")):
        query = row[query_column]
        # Label the instance by its position in the source file, not by
        # position in this slice -- ids have to stay comparable across batches.
        instance_id = int(idx) if isinstance(idx, (int,)) else pos
        gold = row[gold_column] if has_gold else None
        with log.instance(instance_id=instance_id, query=query,
                          gold_objective=gold) as rec:
            category, output_model = None, None
            try:
                category, output_model = Process_Input(query)
                rec.set(pred_type=category, model_output=output_model,
                        code_output=_extract_gurobipy_code(output_model))
            except Exception as e:
                print(f"[{instance_id}] failed: {type(e).__name__}: {e}")
                rec.fail(e)                      # status -> "failed" in the log
        results.append({
            "Category": category,
            "Original_Query": query,
            "Output": output_model,
        })

    return pd.DataFrame(results)
'''

PROCESS_INPUT_AIR = '''
def Process_Input(query):
    with lx.stage("classification"):
        category_original = Problemtype(query)
    print(f"Problem type classification finished, it belongs to {category_original}.")

    # NOTE: the original code left Type / output_model unbound when the
    # category did not contain "Sales-Based Linear Programming", raising
    # UnboundLocalError instead of recording a failure. The else branch below
    # makes that case explicit.
    if "Sales-Based Linear Programming" in category_original or "Sales-Based" in category_original:
        print("Processing AirNRM queries")
        if "flow conservation constraints" in query or "flow conservation constraint" in query:
            print('----------Flow Constraints----------')
            print("Recommend Optimal Flights With Flow Conervation Constraints")
            with lx.stage("modeling"):
                output_model = ProcessPolicyFlow(query)
            Type = "Policy_Flow"
        else:
            print('----------CA----------')
            print("Only Develop Mathematic Formulations. No Recommendation for Flights.")
            with lx.stage("modeling"):
                output_model = ProcessCA(query)
            Type = "CA"
    else:
        raise ValueError(
            f"unroutable problem type: {category_original!r}")

    return Type, output_model
'''

RUN_TEST_BENCH = '''
import pandas as pd
from langchain_core.messages import HumanMessage
from typing import List


def build_llm(model: str = None, temperature: float = None, role: str = "modeler"):
    """
    Kept as a shim so existing call sites keep working, but the model is now
    chosen by CFG (exp_config.yaml:model_profiles), not by editing which of
    three commented-out definitions is active. Switch models by changing
    model_profile= in the configuration cell.
    """
    return lx.build_llm(CFG, role)


def run_test(df: pd.DataFrame, llm, log=LOG, cfg=CFG) -> List[str]:
    """
    Single-call baseline, instrumented identically to the full pipeline so the
    cost comparison is apples-to-apples (one LLM call per instance here vs.
    several in LEAN-LLM-OPT).
    """
    result = []

    for row_idx, row in df.iterrows():
        query = row['Query']
        prompt = f"""
Below is an operations research question. Build a mathematical model and corresponding python code using 'gurobipy' that appropriately addresses the question.

# Question
{query}

# Response
        """
        messages = [HumanMessage(content=prompt)]

        with log.instance(instance_id=int(row_idx), query=query) as rec:
            response_content = None
            try:
                with lx.stage("single_call"):
                    response = lx.call_with_retry(llm.invoke, cfg, messages)
                response_content = response.content
                print(response_content)
                rec.set(model_output=response_content)
            except Exception as e:
                print(f"Error processing query at Index {row_idx}: {e}")
                rec.fail(e)                      # status -> "failed" in the log
                response_content = f"Error: {e}"
                rec.set(model_output=None)
        result.append(response_content)

    return result


llm1 = build_llm()
'''


# --------------------------------------------------------------------------- #
# per-notebook recipes
# --------------------------------------------------------------------------- #

def patch_air_nrm_41(check):
    p = Patcher(_orig("LEAN_LLM_OPT_4.1_Air_NRM.ipynb"),
                "gpt-4.1", "LEAN-LLM-OPT", "Air-NRM-CA")
    p.insert_config(0)
    p.patch_air_nrm_paths(want=8)
    p.patch_models(want=4)
    p.patch_classification_docs()
    p.patch_embeddings(want=5)
    p.patch_retrievers({
        2: ["refdata"],
        8: ["air_flight", "air_flight", "air_demand", "air_examples"],
        10: ["air_flight", "air_flight", "air_demand", "air_examples"],
    })
    p.patch_agents(want=3)
    p.patch_process_input(12)
    p.patch_batch_loop(14)
    p.patch_to_csv()
    p.patch_read_back()
    p.patch_gurobi_import()
    p.append_close_cell()
    p.validate()
    return p, p.write(check)


def patch_air_nrm_oss(check):
    p = Patcher(_orig("LEAN_LLM_OPT_gpt_oss_20b_Air_NRM.ipynb"),
                "gpt-oss-20b", "LEAN-LLM-OPT", "Air-NRM-CA")
    p.append_config(0)
    p.patch_air_nrm_paths(want=7)
    p.sub_all(
        r'def build_llm\(model: str = "gpt-oss:20b", temperature: float = 0\.0\)'
        r'\s*->\s*ChatOllama:\s*\n(?:.*?\n)*?\s*\)\n',
        'def build_llm(model: str = None, temperature: float = None,\n'
        '              role: str = "modeler"):\n'
        '    """Model choice now comes from CFG (exp_config.yaml)."""\n'
        '    return lx.build_llm(CFG, role)\n',
        1, "build_llm shim")
    p.patch_classification_docs()
    p.patch_embeddings(want=1)
    p.patch_retrievers({
        2: ["refdata"],
        8: ["air_flight", "air_flight", "air_demand", "air_examples"],
        10: ["air_flight", "air_flight", "air_demand", "air_examples"],
    })
    p.patch_agents(want=(0, 1, 2, 3))
    p.patch_process_input(12)
    p.patch_batch_loop(14)
    p.patch_to_csv()
    p.patch_read_back()
    p.patch_gurobi_import()
    p.append_close_cell()
    p.validate()
    return p, p.write(check)


def patch_large_41(check):
    p = Patcher(_orig("LEAN_LLM_OPT_4.1_Large-scale-or.ipynb"),
                "gpt-4.1", "LEAN-LLM-OPT", "Large-Scale-OR")
    p.insert_config(1)
    p.patch_models(want=9)
    p.patch_classification_docs()
    p.patch_embeddings(want=13)
    p.patch_retrievers({
        4:  ["refdata"],
        7:  ["examples_nrm", "data_nrm"],
        9:  ["examples_ra", "data_ra"],
        11: ["examples_tp", "data_tp"],
        13: ["examples_ap", "data_ap"],
        15: ["examples_flp", "data_flp"],
        19: ["examples_others_nocsv"],
    })
    p.patch_agents(want=7)
    p.replace_cell(23, RUN_TEST_LARGE, "def run_test")
    p.patch_to_csv()
    p.patch_read_back()
    p.patch_gurobi_import()
    p.append_close_cell()
    p.validate()
    return p, p.write(check)


def patch_large_oss(check):
    p = Patcher(_orig("LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or.ipynb"),
                "gpt-oss-20b", "LEAN-LLM-OPT", "Large-Scale-OR")
    p.append_config(0)
    p.sub_all(
        r'def build_llm\(model: str = "gpt-oss:20b", temperature: float = 0\.0\)'
        r'\s*->\s*ChatOllama:\s*\n(?:.*?\n)*?\s*\)\n',
        'def build_llm(model: str = None, temperature: float = None,\n'
        '              role: str = "modeler"):\n'
        '    """Model choice now comes from CFG (exp_config.yaml)."""\n'
        '    return lx.build_llm(CFG, role)\n',
        1, "build_llm shim")
    p.patch_classification_docs()
    p.patch_oss_classification(want=1)
    p.patch_embeddings(want=2)
    p.patch_retrievers({
        10: ["oss_data_nrm"],
        12: ["oss_data_ra"],
        14: ["oss_data_ap"],
        16: ["oss_data_flp"],
        22: ["oss_examples_nocsv"],
    })
    p.patch_agents(want=(0, 1, 2, 3, 4, 5, 6))
    p.replace_cell(24, RUN_TEST_OSS_LARGE + "\n\n" +
                   _tail_of_cell(p, 24, "def read_and_combine_csvs"),
                   "def run_test")
    p.patch_to_csv()
    p.patch_read_back()
    p.patch_gurobi_import()
    p.append_close_cell()
    p.validate()
    return p, p.write(check)


def _tail_of_cell(p: Patcher, cell: int, marker: str) -> str:
    """Keep the helper functions that live below run_test in the same cell."""
    s = p.src(cell)
    idx = s.find(marker)
    if idx < 0:
        raise AssertionError(f"marker {marker!r} not found in cell {cell}")
    return s[idx:]


def patch_ablation(path_name, method, check, retrievers=None, air_nrm_paths=0):
    p = Patcher(_orig(path_name), "gpt-4.1", method, "Air-NRM-CA")
    p.insert_config(4)
    p.patch_air_nrm_paths(want=air_nrm_paths)
    p.patch_models()
    p.patch_classification_docs()
    p.patch_embeddings()
    if retrievers:
        p.patch_retrievers(retrievers)
    p.patch_agents()
    # Process_Input / Batch cells are located by content, not by index
    for i, c in enumerate(p.nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = p.src(i)
        if "def Process_Input" in s:
            p.patch_process_input(i)
        if "def Batch_Process_Queries" in s:
            p.patch_batch_loop(i)
    p.patch_to_csv()
    p.patch_read_back()
    p.patch_gurobi_import()
    p.append_close_cell()
    p.validate()
    return p, p.write(check)


def patch_benchmark(check):
    p = Patcher(_orig("Benchmark_Base_Model_Small_Scale.ipynb"),
                "gpt-oss-20b", "Base-SingleCall", "MAMO-complex")
    p.append_config(0)
    p.replace_cell(2, RUN_TEST_BENCH, "def run_test")
    p.patch_classification_docs()
    p.patch_embeddings(want=(0, 1))
    p.patch_to_csv()
    p.patch_read_back()
    p.patch_gurobi_import()
    p.append_close_cell()
    p.validate()
    return p, p.write(check)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="dry run: validate rules, write nothing")
    ap.add_argument("--only", default=None, help="substring filter")
    args = ap.parse_args()

    jobs = [
        ("LEAN_LLM_OPT_4.1_Air_NRM", patch_air_nrm_41),
        ("LEAN_LLM_OPT_gpt_oss_20b_Air_NRM", patch_air_nrm_oss),
        ("LEAN_LLM_OPT_4.1_Large-scale-or", patch_large_41),
        ("LEAN_LLM_OPT_gpt_oss_20b_Large-scale-or", patch_large_oss),
        ("Ablation_Study_Air_NRM_RAG_Only",
         lambda c: patch_ablation(
             "Ablation_Study_Air_NRM_RAG_Only.ipynb", "Abl-RAGOnly", c,
             {6: ["refdata"],
              9: ["air_flight", "air_flight", "air_demand", "air_examples"],
              10: ["air_flight", "air_flight", "air_demand", "air_examples"]},
             air_nrm_paths=6)),
        ("Ablation_Study_Air_NRM_Few-shot_Only",
         lambda c: patch_ablation(
             "Ablation_Study_Air_NRM_Few-shot_Only.ipynb", "Abl-FewShotOnly", c,
             {5: ["refdata"]}, air_nrm_paths=15)),
        ("Benchmark_Base_Model_Small_Scale", patch_benchmark),
    ]

    failures = 0
    for name, fn in jobs:
        if args.only and args.only not in name:
            continue
        print(f"\n=== {name} ===")
        try:
            p, out = fn(args.check)
            for line in p.report:
                print(line)
            print(f"  -> {out.name}" + (" (dry run)" if args.check else ""))
        except Exception as e:                        # noqa: BLE001
            failures += 1
            print(f"  FAILED: {type(e).__name__}: {e}")

    print(f"\n{len(jobs) - failures}/{len(jobs)} notebooks patched"
          + (" (dry run)" if args.check else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
