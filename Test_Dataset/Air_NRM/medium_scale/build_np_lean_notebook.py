#!/usr/bin/env python3
"""Build/check the dedicated small-scale Air-NRM NP notebook."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "LEAN_LLM_OPT_4.1_Air_NRM.ipynb"
OUTPUT = HERE / "LEAN_LLM_OPT_4.1_Air_NRM_NP_SmallScale.ipynb"


FLOW_CELL = '''# SMALL-SCALE NP/FLOW: retriever-backed data tools and stable integer keys.
from Test_Dataset.Air_NRM.small_scale.np_lean_workflow import install_np_workflow

_NP_WORKFLOW = install_np_workflow(globals())
globals().update(_NP_WORKFLOW)
'''


ROUTING_CELL = '''def Process_Input(query):
    with lx.stage("classification"):
        category_original = Problemtype(query)
    print(f"Problem type classification finished: {category_original}.")
    if "Sales-Based Linear Programming" not in category_original:
        raise ValueError(f"unroutable problem type: {category_original!r}")
    if "flow conservation constraint" not in query.lower():
        raise ValueError("this NP-only notebook requires a flow-conservation query")
    with lx.stage("modeling"):
        output_model = ProcessPolicyFlow(query)
    return "Policy_Flow", output_model
'''


EXTRACTION_CELL = '''import re as _re
from tqdm import tqdm


_FENCE_RE = _re.compile(r"```(?:python|py)\\s*\\n(.*?)```", _re.I | _re.S)


def _extract_gurobipy_code(text):
    blocks = _FENCE_RE.findall(str(text or ""))
    if len(blocks) != 1 or "gurobipy" not in blocks[0]:
        raise ValueError("expected one complete gurobipy code block")
    return blocks[0].strip()


def Batch_Process_Queries(df, query_column="Query", log=LOG, cfg=CFG,
                          gold_column="Label-objective"):
    results = []
    has_gold = gold_column in df.columns
    for pos, (idx, row) in enumerate(tqdm(list(df.iterrows()), desc="Processing Queries")):
        query = row[query_column]
        instance_id = int(idx) if isinstance(idx, int) else pos
        gold = row[gold_column] if has_gold else None
        category, raw_model_output = None, None
        with log.instance(instance_id=instance_id, query=query,
                          gold_objective=gold) as rec:
            try:
                NP_SET_CURRENT_QUERY(query)
                category, raw_model_output = Process_Input(query)
                executable_code = _extract_gurobipy_code(raw_model_output)
                rec.set(pred_type=category, model_output=raw_model_output,
                        code_output=executable_code,
                        code_assembly="validated_direct_LLM_code")
            except Exception as exc:
                print(f"[{instance_id}] failed: {type(exc).__name__}: {exc}")
                rec.fail(exc)
        results.append({"Category": category, "Original_Query": query,
                        "Output": raw_model_output})
    return pd.DataFrame(results)
'''


RUN_CELL = '''# Small-scale NP-only full run.
df = pd.read_csv("Test_Dataset/Air_NRM/small_scale/query_NP_Flow.csv")
result_df = Batch_Process_Queries(df)
result_df.to_csv(LOG.dir / "answer_NP_Flow_smallscale_GPT4.1.csv", index=False)
'''


def source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def build() -> dict:
    nb = copy.deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    config = "".join(nb["cells"][0]["source"])
    config = config.replace("dataset='Air-NRM-CA'", "dataset='Air-NRM-NP'")
    config = config.replace("load_dotenv(override=True)", "load_dotenv(override=False)")
    config = config.replace(
        "# override=True: the .env file wins over anything already in the\n"
        "# environment. Without it, a stale key left in os.environ by an earlier\n"
        "# run silently shadows the value you just edited into .env.\n",
        "# Keep a key entered with run_np_lean.py --prompt-key ahead of .env.\n",
    )
    if "Air-NRM-NP" not in config:
        raise ValueError("failed to select the small-scale NP dataset")
    nb["cells"][0]["source"] = source(config)
    nb["cells"][8]["source"] = source(FLOW_CELL)
    nb["cells"][10]["source"] = source("# CA workflow omitted from this NP-only notebook.\n")
    nb["cells"][12]["source"] = source(ROUTING_CELL)
    nb["cells"][14]["source"] = source(EXTRACTION_CELL)
    nb["cells"][15]["source"] = source("# Generated code is scored after the run.\n")
    nb["cells"][16]["source"] = source("### Run small-scale NP / Flow\n")
    nb["cells"][17]["source"] = source(RUN_CELL)
    for idx in range(18, len(nb["cells"])):
        nb["cells"][idx]["source"] = source(
            "# Small-scale NP execution is defined above; no CA fallback is present.\n"
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
    parser.add_argument("--check", action="store_true")
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
