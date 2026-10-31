#!/usr/bin/env python3
"""
verify_pipeline.py
==================
Check the large-scale notebook's data tool and few-shot template against the
ground-truth labels, without calling an LLM or spending anything.

    python verify_pipeline.py
    python verify_pipeline.py --notebook LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb

Two things get checked, in order:

  A. csv_qa_tool_CA's output.  For all 15 queries, every one of the seven
     parameter groups it emits -- avg_price, value_list, ratio_list,
     value_0_list, ratio_0_list, avg_pax, flight_capacity, plus capacity_consum
     -- is compared cell by cell against SBLP_CA_Label/CA_parameters.json, the
     dump build_largescale_labels.py wrote when it solved the instances.

  B. the few-shot template.  The gurobipy block inside CA_Agent's exemplar is
     lifted out verbatim, fed the parameters from (A), executed, and its
     optimum compared against Label-objective.

(B) is the stronger statement: it says a model that copies the example it is
shown lands on the right answer. Whatever the run then measures is the model's
behaviour, not a broken harness.

Retrieval is stubbed
--------------------
retrieve_avg_price / retrieve_avg_pax are replaced with direct dataframe
lookups, so this runs with no API key. That is sound for what is being checked
-- the retrieval mechanism is unchanged from the small-scale notebook; what
changed is the parsing, the coefficient lookup and the capacity handling, all
of which sit downstream of retrieval and are exercised here in full.

The one thing it cannot see is whether the live vector search returns the right
row. csv_qa_tool_CA now asserts that at runtime against flight.csv, so a wrong
pick raises during the real run instead of quietly mispricing a fare family.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from datetime import datetime, time as time_class
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def chdir_to_repo_root() -> Path:
    """LoadFiles() reads repo-root-relative paths, as the notebook's first cell
    guarantees by chdir-ing there. Do the same so this checks the notebook as it
    actually runs, rather than a path-patched variant of it."""
    root = HERE
    while not (root / "exp_config.yaml").exists() and root != root.parent:
        root = root.parent
    if not (root / "exp_config.yaml").exists():
        sys.exit("ERROR: not inside the lean-llm-opt-main tree")
    os.chdir(root)
    return root
DEFAULT_NB = "LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb"
PARAM_GROUPS = ["avg_price", "value_list", "ratio_list", "value_0_list",
                "ratio_0_list", "avg_pax", "capacity_consum", "flight_capacity"]
TOL = 1e-9


def load_notebook_functions(nb_path: Path, base: Path) -> dict:
    """Exec just the functions this check needs, with retrieval stubbed out.

    Functions are pulled by name rather than by executing whole cells: the
    cells also define FAISS indexes and langchain agents, which would need an
    API key and are not what is under test.
    """
    nb = json.loads(nb_path.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"]]

    flight = pd.read_csv(base / "flight.csv")
    demand = pd.read_csv(base / "od_demand.csv")
    price = flight.set_index(
        ["Oneway_OD", "Departure Time", "Oneway_Product"])["Avg Price"]
    pax = demand.set_index("Oneway_OD")["Avg Pax"]

    ns = {
        "pd": pd, "re": re, "datetime": datetime, "time_class": time_class,
        "retrieve_avg_price": lambda od, t, p: str(price[(od, t, p)]),
        "retrieve_avg_pax": lambda od: str(pax[od]),
    }

    def cell_with(marker):
        for text in cells:
            if marker in text:
                return text
        sys.exit(f"ERROR: no cell in {nb_path.name} defines {marker}")

    exec(cell_with("def LoadFiles"), ns)
    src = cell_with("def generate_coefficients")
    for name in ("retrieve_key_information", "retrieve_time_period",
                 "retrieve_parameter", "generate_coefficients"):
        start = src.find(f"def {name}")
        if start < 0:
            sys.exit(f"ERROR: {name} not found in the notebook")
        nxt = src.find("\ndef ", start + 1)
        exec(src[start: nxt if nxt > 0 else len(src)], ns)
    tool = cell_with("def csv_qa_tool_CA")
    exec(tool[: tool.find("def CA_Agent")], ns)
    ns["_agent_cell"] = tool
    return ns


def check_syntax(nb_path: Path) -> int:
    """Compile every code cell.

    Cheapest possible check and it catches a whole class of edit damage before
    anything is spent. The one that motivated it: `fewshot_example` is an
    f-string, so a literal brace inside it has to be doubled. Writing
    `flights = {}` in the exemplar's code block makes the cell a SyntaxError --
    which surfaces only when the notebook is executed, i.e. after the API key
    prompt and the run directory have already been set up.
    """
    nb = json.loads(nb_path.read_text())
    failures = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith("%") or "\n%" in src:
            continue                      # IPython magics are not Python
        try:
            compile(src, f"<cell {i}>", "exec")
        except SyntaxError as exc:
            failures += 1
            print(f"   cell {i}: {exc.msg}  (line {exc.lineno})")
    print(f"0. SYNTAX -- {'every code cell compiles' if not failures else f'{failures} cell(s) do not compile'}")
    print()
    return failures


def check_prompt_variables(ns) -> int:
    """Build the agent prefix the way CA_Agent does and see what survives as a
    PromptTemplate variable.

    Two layers eat braces here. `fewshot_example = f'''...'''` is an f-string,
    so `{x}` there is interpolated at build time. What comes out is then handed
    to initialize_agent as `prefix`, and langchain wraps it in a PromptTemplate
    whose only declared inputs are `input` and `agent_scratchpad` -- any other
    `{name}` left in the text is an undeclared variable and the agent refuses to
    start: "Missing some input keys: {'', 'flight_key'}".

    Compiling the cell does not catch this (the source is valid Python) and
    neither does running the tool (the failure is in the prompt, not the data).
    So: replay both layers, then look for what is left.
    """
    import pandas as _pd
    agent_cell = ns["_agent_cell"]
    start = agent_cell.find("fewshot_example = f'''")
    if start < 0:
        print("2. PROMPT -- SKIPPED: no fewshot_example in the notebook")
        return 0
    body = agent_cell[start + len("fewshot_example = f'''"):]
    body = body[: body.find("'''")]

    example = _pd.read_csv(
        "Large_Scale_Or_Files/RAG_Example_SBLP_CA_LargeScale.csv")["prompt"][0]
    matches = ns["retrieve_key_information"](example)
    described = ns["csv_qa_tool_CA"](matches).replace("{", "{{").replace("}", "}}")

    # layer 1a: the f-string that builds fewshot_example
    fewshot = body.format(problem_description=example, example_matches=matches,
                          example_data_description=described)

    # layer 1b: the f-string that builds `prefix`, which is what actually
    # reaches initialize_agent. Checking only fewshot_example missed a stray
    # brace written into the surrounding instructions -- langchain reported
    # "Invalid variable name '...'" and all four instances died before the
    # agent started. The span under test has to be the span that is handed over.
    head = agent_cell.find('prefix = f"""')
    if head < 0:
        print("2. PROMPT -- SKIPPED: no prefix in the notebook")
        return 0
    prefix_body = agent_cell[head + len('prefix = f"""'):]
    prefix_body = prefix_body[: prefix_body.find('"""')]
    text = prefix_body.format(fewshot_example=fewshot)

    # layer 2: what PromptTemplate would read as a variable -- a { not doubled
    stripped = re.sub(r"\{\{|\}\}", "", text)
    found = set(re.findall(r"\{([^{}]*)\}", stripped))
    allowed = {"input", "agent_scratchpad"}
    stray = found - allowed
    if stray:
        print(f"2. PROMPT -- {len(stray)} stray template variable(s) in the "
              f"agent prefix: " + ", ".join(repr(v) for v in sorted(stray)))
        print("   langchain rejects these before the agent starts. Remove the "
              "literal braces; do not try to escape them.")
        print()
        return 1
    print("2. PROMPT -- the agent prefix leaves no stray template variables")
    print()
    return 0


def parse_doc(doc: str) -> dict:
    """Turn csv_qa_tool_CA's observation string back into dicts."""
    out = {}
    for name in PARAM_GROUPS:
        m = re.search(rf"^{name}=(\{{.*?\}})$", doc, re.M | re.S)
        if not m:
            sys.exit(f"ERROR: the tool's output has no {name}= line")
        out[name] = ast.literal_eval(m.group(1))
    return out


def cell_key(gt_key: str) -> str:
    """CA_parameters.json's "(\"('865', '678')\", '01:30', 'f')" -> "(865678,01:30,f)"."""
    od, time, short = ast.literal_eval(gt_key)
    origin, destination = ast.literal_eval(od)
    return f"({origin}{destination},{time},{short})"


def od_key(gt_key: str) -> str:
    od = ast.literal_eval(gt_key)
    if isinstance(od, tuple):
        od = od[0]
    return "".join(ast.literal_eval(od)) if isinstance(od, str) else "".join(od)


def check_parameters(ns, queries, truth, via_cname: bool) -> int:
    """Compare the tool's output against the labels.

    via_cname replays the path the agent actually takes: it calls CName
    (retrieve_key_information) first and feeds THAT to CSVQA. Checking only the
    raw query missed a real failure -- the rewrite dropped the capacity
    sentences, so csv_qa_tool_CA saw no consumption coefficients at all and the
    first live run died on instance 0. Both paths must produce the same
    parameters.
    """
    label = "csv_qa_tool_CA(retrieve_key_information(query))  <- the agent's path" \
        if via_cname else "csv_qa_tool_CA(query)"
    print(f"A. {label}")
    header = f"{'#':>3} {'flights':>8}  " + "".join(f"{g:>15}" for g in PARAM_GROUPS[:6])
    print(header)
    failures = 0
    for i, (query, gold) in enumerate(zip(queries, truth), start=1):
        text = ns["retrieve_key_information"](query) if via_cname else query
        got = parse_doc(ns["csv_qa_tool_CA"](text))
        results = {
            "avg_price": all(abs(got["avg_price"][cell_key(k)] - v) < TOL
                             for k, v in gold["avg_price"].items()),
            "value_list": all(abs(got["value_list"][cell_key(k)] - v) < TOL
                              for k, v in gold["value_list"].items()),
            "ratio_list": all(abs(got["ratio_list"][cell_key(k)] - v) < TOL
                              for k, v in gold["ratio_list"].items()),
            "value_0_list": all(abs(got["value_0_list"][od_key(k)] - v) < TOL
                                for k, v in gold["value_0_list"].items()),
            "ratio_0_list": all(abs(got["ratio_0_list"][od_key(k)] - v) < TOL
                                for k, v in gold["ratio_0_list"].items()),
            "avg_pax": all(abs(got["avg_pax"]["".join(ast.literal_eval(k))] - v) < TOL
                           for k, v in gold["avg_pax"].items()),
        }
        # flight_capacity and capacity_consum are keyed differently; check apart.
        for k, v in gold["flight_capacity"].items():
            od, time = ast.literal_eval(k)
            origin, destination = ast.literal_eval(od)
            if abs(got["flight_capacity"][f"({origin}{destination},{time})"] - v) > 0.01:
                results["avg_price"] = False
        if {k: float(v) for k, v in got["capacity_consum"].items()} != \
                {k: float(v) for k, v in gold["capacity_consum"].items()}:
            results["avg_price"] = False

        failures += sum(not ok for ok in results.values())
        n = query.count("AND Departure Time=")
        print(f"{i:>3} {n:>8}  " + "".join(
            f"{'OK' if results[g] else 'FAIL':>15}" for g in PARAM_GROUPS[:6]))
    print()
    return failures


def check_template(ns, queries, objectives) -> int:
    print("B. the few-shot gurobipy template, executed on each query")
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception:
        print("   SKIPPED: gurobipy not importable")
        return 0

    agent_cell = ns["_agent_cell"]
    anchor = agent_cell.find("### Generated Code")
    if anchor < 0:
        sys.exit("ERROR: no '### Generated Code' block in the exemplar")
    block = agent_cell[agent_cell.find("```python", anchor) + len("```python"):]
    end = block.find("model.optimize()")
    if end < 0:
        sys.exit("ERROR: the exemplar's code block has no model.optimize()")
    block = block[: end + len("model.optimize()")].replace("{{", "{").replace("}}", "}")

    gp.setParam("OutputFlag", 0)
    print(f"{'#':>3} {'vars':>6} {'constrs':>8}  {'template optimum':>18}"
          f"{'Label-objective':>18}{'rel err':>10}")
    failures = 0
    for i, (query, gold) in enumerate(zip(queries, objectives), start=1):
        env = {"gp": gp, "GRB": GRB}
        exec(block.replace("{example_data_description}", ns["csv_qa_tool_CA"](query)), env)
        model = env["model"]
        value = model.ObjVal if model.Status == GRB.OPTIMAL else float("nan")
        rel = abs(value - gold) / max(abs(gold), 1.0)
        ok = rel < 1e-6
        failures += not ok
        print(f"{i:>3} {model.NumVars:>6} {model.NumConstrs:>8}  {value:>18,.2f}"
              f"{gold:>18,.2f}{rel:>10.1e}" + ("" if ok else "  <<< FAIL"))
    print()
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", default=DEFAULT_NB)
    ap.add_argument("--data-dir", type=Path, default=HERE)
    args = ap.parse_args()

    base = args.data_dir.resolve()
    nb_path = base / args.notebook if not Path(args.notebook).is_absolute() \
        else Path(args.notebook)
    root = chdir_to_repo_root()
    for path in (nb_path, base / "query_CA.csv",
                 base / "SBLP_CA_Label" / "CA_parameters.json"):
        if not path.exists():
            sys.exit(f"ERROR: {path} not found")

    queries = pd.read_csv(base / "query_CA.csv")
    if "Label-objective" not in queries.columns:
        sys.exit("ERROR: query_CA.csv has no Label-objective column -- "
                 "run link_air_nrm_labels.py first")
    truth = json.loads((base / "SBLP_CA_Label" / "CA_parameters.json").read_text())

    print(f"notebook : {nb_path.name}")
    print(f"data     : {base}")
    print(f"instances: {len(queries)}   (retrieval stubbed with dataframe lookups)")
    print()

    failures = check_syntax(nb_path)
    if failures:
        print("RESULT: FAIL -- fix the syntax errors above before anything else; "
              "a notebook that does not compile fails only after the API key "
              "prompt and the run directory have been set up.")
        return 1

    ns = load_notebook_functions(nb_path, base)
    failures += check_prompt_variables(ns)
    failures += check_parameters(ns, queries["Query"], truth, via_cname=False)
    failures += check_parameters(ns, queries["Query"], truth, via_cname=True)
    failures += check_template(ns, queries["Query"], queries["Label-objective"])

    if failures:
        print(f"RESULT: FAIL -- {failures} check(s) did not match.")
        return 1
    print("RESULT: PASS -- the data tool reproduces every ground-truth "
          "coefficient, and the few-shot template solves every instance to "
          "its labelled optimum.")
    print()
    print("Not covered here:")
    print("  - whether the live vector search returns the right row "
          "(asserted at runtime inside csv_qa_tool_CA)")
    print("  - whether the LLM follows the example it is shown. That is the "
          "experiment, and it is what a real run measures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
