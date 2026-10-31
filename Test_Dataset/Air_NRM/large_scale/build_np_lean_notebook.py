#!/usr/bin/env python3
"""Build/check the independent large-scale Air-NRM NP LEAN notebook."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "LEAN_LLM_OPT_4.1_Air_NRM_LargeScale.ipynb"
OUTPUT = HERE / "LEAN_LLM_OPT_4.1_Air_NRM_NP_LargeScale.ipynb"


FLOW_CELL = '''# LARGE-SCALE NP/FLOW: deterministic five-file data assembly plus LEAN tools.
from Test_Dataset.Air_NRM.large_scale.np_lean_workflow import install_np_workflow

_NP_WORKFLOW = install_np_workflow(globals())
globals().update(_NP_WORKFLOW)
'''


ROUTING_CELL = '''def _is_airnrm_category(category):
    # Local models vary capitalization, hyphen style and the SBLP class name.
    normalized = re.sub(r"[\\u2010-\\u2015\\u2212]", "-", str(category)).casefold()
    return bool(re.search(
        r"\\bsales[-\\s]+based\\s+linear\\s+programming\\b|\\bsblp\\b|"
        r"\\b(?:network\\s+)?revenue\\s+management\\b", normalized
    ))


def Process_Input(query):
    with lx.stage("classification"):
        category_original = Problemtype(query)
    print(f"Problem type classification finished, it belongs to {category_original}.")
    if not _is_airnrm_category(category_original):
        raise ValueError(f"unroutable problem type: {category_original!r}")
    if "flow conservation constraint" not in query.lower():
        raise ValueError("this NP-only notebook requires a flow-conservation query")
    print("----------Large-scale NP / Flow Constraints----------")
    with lx.stage("modeling"):
        output_model = ProcessPolicyFlow(query)
    return "Policy_Flow", output_model
'''


EXTRACTION_CELL = '''import ast as _ast
import re as _re
from tqdm import tqdm


_FENCE_RE = _re.compile(
    r"```([A-Za-z0-9_+-]*)\\s*\\n(.*?)```", _re.I | _re.S
)
_PARAMETER_NAMES = (
    "products", "flight_info", "avg_price", "value_list", "ratio_list",
    "value_0_list", "ratio_0_list", "avg_pax", "capacity_consum",
    "flight_capacity", "option_num",
)


def _extract_gurobipy_code(text, query=None):
    """Extract and validate LLM code verbatim; never repair or inject data."""
    if not text:
        raise ValueError("LLM returned no answer")
    objective_heading = str(text).find("### Objective Function")
    constraints_heading = str(text).find("### Constraints")
    code_heading = str(text).find("### Generated Code")
    fence_start = str(text).find("```", code_heading if code_heading >= 0 else 0)
    if not 0 <= objective_heading < constraints_heading < code_heading < fence_start:
        raise ValueError(
            "LLM answer must contain Objective Function, Constraints, then Generated Code"
        )
    if "### Retrieved Information" in str(text):
        raise ValueError("LLM answer duplicated parameters outside Generated Code")
    blocks = [
        (match.group(1).lower(), match.group(2).strip(), match.start())
        for match in _FENCE_RE.finditer(str(text))
    ]
    python_blocks = [block for block in blocks if block[0] in ("python", "py")]
    if len(python_blocks) != 1:
        raise ValueError(
            f"expected exactly one tagged Python code block, got {len(python_blocks)}"
        )
    _, code, python_fence_start = python_blocks[0]
    if python_fence_start <= code_heading:
        raise ValueError("tagged Python block is not under ### Generated Code")
    if "gurobipy" not in code or "Model(" not in code:
        raise ValueError("LLM code contains no Gurobi model")
    if "..." in code:
        raise ValueError("LLM code contains an ellipsis placeholder")
    tree = _ast.parse(code)
    if any(isinstance(node, _ast.Constant) and node.value is Ellipsis
           for node in _ast.walk(tree)):
        raise ValueError("LLM code contains an ellipsis placeholder")

    assignments = {}
    for node in tree.body:
        if (isinstance(node, _ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], _ast.Name)
                and node.targets[0].id in _PARAMETER_NAMES):
            try:
                assignments[node.targets[0].id] = _ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"parameter {node.targets[0].id} is not a complete literal"
                ) from exc
    missing = [name for name in _PARAMETER_NAMES if name not in assignments]
    if missing:
        raise ValueError(f"LLM code is missing parameters: {missing}")

    # Verification only: compare with source data without changing the LLM code.
    if query is not None:
        expected = NP_DATA.assemble(query)
        different = [name for name in _PARAMETER_NAMES
                     if assignments[name] != expected[name]]
        if different:
            raise ValueError(f"LLM code changed parameters: {different}")
    return code


def Batch_Process_Queries(df, query_column='Query', log=LOG, cfg=CFG,
                          gold_column='Label-objective'):
    """Keep the direct LLM answer and directly extracted executable code."""
    results = []
    has_gold = gold_column in df.columns
    for pos, (idx, row) in enumerate(
            tqdm(list(df.iterrows()), desc="Processing Queries")):
        query = row[query_column]
        instance_id = int(idx) if isinstance(idx, int) else pos
        gold = row[gold_column] if has_gold else None
        category, raw_model_output = None, None
        with log.instance(instance_id=instance_id, query=query,
                          gold_objective=gold) as rec:
            try:
                NP_SET_CURRENT_QUERY(query)
                category, raw_model_output = Process_Input(query)
                executable_code = _extract_gurobipy_code(raw_model_output, query)
                rec.set(
                    pred_type=category,
                    model_output=raw_model_output,
                    code_output=executable_code,
                    code_assembly="direct_LLM_code_extraction",
                )
            except Exception as exc:
                print(f"[{instance_id}] failed: {type(exc).__name__}: {exc}")
                rec.fail(exc)
        results.append({
            "Category": category,
            "Original_Query": query,
            "Output": raw_model_output,
        })
    return pd.DataFrame(results)
'''


RUN_CELL = '''# NP-only full-run cell. run_np_lean.py stops before this cell and selects rows safely.
df = pd.read_csv('Test_Dataset/Air_NRM/large_scale/query_NP_Flow.csv')
result_df = Batch_Process_Queries(df)
result_df.to_csv(LOG.dir / "answer_NP_Flow_largescale_GPT4.1.csv", index=False)

code_df = pd.DataFrame(
    [_extract_gurobipy_code(text, query)
     for text, query in zip(result_df['Output'], result_df['Original_Query'])],
    columns=['Code'],
)
code_df.to_csv(LOG.dir / "code_NP_Flow_largescale_GPT4.1.csv", index=False)
combined_df = pd.concat([result_df, code_df], axis=1)
combined_df.to_csv(LOG.dir / "final_NP_Flow_largescale_GPT4.1.csv", index=False)
'''


def as_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def build() -> dict:
    nb = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    config = "".join(nb["cells"][1]["source"])
    config = config.replace("dataset='Air-NRM-CA-LS'", "dataset='Air-NRM-NP-LS'")
    config = config.replace("load_dotenv(override=True)", "load_dotenv(override=False)")
    config, replacements = re.subn(
        r'repeat_index\s*=\s*int\(os\.environ\.get\("LEANOPT_REPEAT",\s*"0"\)\)',
        "repeat_index=0",
        config,
        count=1,
    )
    if replacements != 1:
        raise ValueError("failed to make repeat_index compatible with run_notebook.py")
    config = config.replace(
        "# override=True: the .env file wins over anything already in the\n"
        "# environment. Without it, a stale key left in os.environ by an earlier\n"
        "# run silently shadows the value you just edited into .env.\n",
        "# Keep a key entered by run_np_lean.py --prompt-key ahead of .env.\n",
    )
    if "Air-NRM-NP-LS" not in config or "repeat_index=0" not in config:
        raise ValueError("failed to replace the notebook dataset")
    nb["cells"][1]["source"] = as_source(config)

    nb["cells"][0]["source"] = as_source(
        "# Independent Air-NRM large-scale NP/Flow LEAN notebook.\n"
        "# Generated by build_np_lean_notebook.py from the maintained large-scale CA setup.\n"
        + "".join(nb["cells"][0]["source"])
    )
    nb["cells"][9]["source"] = as_source(FLOW_CELL)
    nb["cells"][11]["source"] = as_source(
        "# CA-only modeling workflow intentionally omitted from this NP-only notebook.\n"
    )
    nb["cells"][13]["source"] = as_source(ROUTING_CELL)
    nb["cells"][15]["source"] = as_source(EXTRACTION_CELL)
    nb["cells"][18]["source"] = as_source(RUN_CELL)
    for idx in (20, 21, 22):
        nb["cells"][idx]["source"] = as_source(
            "# NP execution is defined in cell 18; no CA or small-scale fallback is present.\n"
        )
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return nb


def serialized() -> str:
    return json.dumps(build(), ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare without writing")
    args = parser.parse_args()
    expected = serialized()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"out of date: {OUTPUT}")
        print(f"OK: {OUTPUT.name} is reproducible and current")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
