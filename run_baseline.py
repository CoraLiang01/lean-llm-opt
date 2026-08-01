#!/usr/bin/env python3
"""
run_baseline.py
===============
Single-call baseline, instrumented identically to LEAN-LLM-OPT so the two can
be put side by side in the cost/accuracy table.

    python run_baseline.py --dataset Large-Scale-OR --profile gpt-4.1 --limit 3
    python run_baseline.py --dataset Large-Scale-OR --profile gpt-5.2
    python run_baseline.py --dataset MAMO-complex   --profile gemini-3-pro
    python run_baseline.py --dataset Large-Scale-OR --profile gpt-4.1 --dry-run

Why this exists
---------------
Referee 2 Q3 asks for LEAN-LLM-OPT's runtime cost *next to* a single-call
baseline. That comparison is only meaningful if both sides are measured the
same way, on the same instances, with the same accounting. Numbers obtained by
hand in a web chat UI cannot supply tokens, calls, cost or latency, and the web
UI is not the same system as the API (different system prompt, possibly tools),
so this script re-runs those baselines through the API.

What "single call" means here
-----------------------------
One LLM call per instance. No classification, no workflow, no retrieval, no
tools. The dataset is inlined into the prompt as text -- that is the whole
point of the baseline: how far does a strong model get when you simply hand it
the problem and the data?

Row budget
----------
Large-Scale-OR instances reference CSVs that can be far larger than any context
window, so rows are truncated. Truncation is never silent: the prompt says how
many rows were dropped, and every instance records rows_total / rows_included /
truncated in the run log, so the paper can state exactly what the baseline saw.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

import leanopt_exp as lx

HERE = Path(__file__).parent


def _check_env() -> None:
    """
    Fail early and clearly when the script is run with the wrong interpreter.
    The usual cause is `python run_baseline.py` picking up conda base instead
    of the environment where the notebook dependencies were installed.
    """
    problems = []
    try:
        import langchain_openai  # noqa: F401
    except Exception as e:                            # noqa: BLE001
        problems.append(f"langchain_openai: {type(e).__name__}: {e}")
    try:
        import langchain_core  # noqa: F401
    except Exception as e:                            # noqa: BLE001
        problems.append(f"langchain_core: {type(e).__name__}: {e}")
    if problems:
        print("This interpreter cannot import the LangChain packages:\n",
              file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(f"\ninterpreter: {sys.executable}\n\n"
              "Run the script with the same Python the notebook kernel uses.\n"
              "In the notebook:  import sys; print(sys.executable)\n"
              "then:             <that path> run_baseline.py ...",
              file=sys.stderr)
        raise SystemExit(2)

PROMPT_TEMPLATE = """\
Below is an operations research question. Build a mathematical model and \
corresponding python code using 'gurobipy' that appropriately addresses the \
question.

# Question
{query}
{data_section}
# Response
"""

DATA_HEADER = """
# Data
The following files are provided. Use the real values from them as the \
parameters of your model; do not invent placeholder numbers.
"""


# --------------------------------------------------------------------------- #
def load_dataset(name: str, cfg_raw: dict) -> pd.DataFrame:
    datasets = cfg_raw.get("datasets", {})
    if name not in datasets:
        raise SystemExit(f"unknown dataset {name!r}; known: {list(datasets)}")
    path = datasets[name]
    if not path:
        raise SystemExit(f"dataset {name!r} has no path in exp_config.yaml")
    p = HERE / path
    if not p.exists():
        raise SystemExit(f"dataset file not found: {p}")
    return pd.read_csv(p)


def build_data_section(dataset_address, max_rows_per_file: int,
                       max_chars: int) -> tuple[str, dict]:
    """
    Inline the referenced CSVs as text. Returns (text, stats).

    stats records exactly what the model was shown, so the truncation can be
    reported rather than hidden.
    """
    stats = {"files": 0, "files_failed": 0, "rows_total": 0,
             "rows_included": 0, "truncated": False, "chars": 0,
             "missing_files": []}
    if not isinstance(dataset_address, str) or not dataset_address.strip():
        return "", stats

    blocks = []
    for raw in dataset_address.strip().splitlines():
        fp = raw.strip()
        if not fp:
            continue
        stats["files"] += 1
        try:
            df = pd.read_csv(HERE / fp)
        except Exception as e:                       # noqa: BLE001
            stats["files_failed"] += 1
            stats["missing_files"].append(f"{fp}: {type(e).__name__}")
            continue

        stats["rows_total"] += len(df)
        shown = df.head(max_rows_per_file)
        stats["rows_included"] += len(shown)
        note = ""
        if len(df) > len(shown):
            stats["truncated"] = True
            note = (f"\n[... {len(df) - len(shown)} further rows omitted; "
                    f"the file has {len(df)} rows in total ...]")

        body = "\n".join(
            ", ".join(f"{c} = {r[c]}" for c in df.columns)
            for _, r in shown.iterrows())
        blocks.append(f"\n## File: {Path(fp).name}\n"
                      f"Columns: {', '.join(map(str, df.columns))}\n"
                      f"{body}{note}\n")

    text = DATA_HEADER + "".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars] + \
            "\n[... data section truncated to fit the prompt budget ...]\n"
        stats["truncated"] = True
    stats["chars"] = len(text)
    return text, stats


def extract_code(text: str) -> str | None:
    if not text:
        return None
    blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    for b in blocks:
        if "gurobipy" in b:
            return b.strip()
    if blocks:
        return blocks[0].strip()
    return text.strip() if "gurobipy" in text else None


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="key from exp_config.yaml:datasets")
    ap.add_argument("--profile", required=True,
                    help="key from exp_config.yaml:model_profiles")
    ap.add_argument("--method", default="Base-SingleCall")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N instances (use 2-3 to smoke test)")
    ap.add_argument("--rows", default=None,
                    help="comma-separated row indices, e.g. 15,45,81 -- use "
                         "this to run exactly the same instances as the "
                         "notebook so the two runs are directly comparable")
    ap.add_argument("--repeat-index", type=int, default=0)
    ap.add_argument("--max-rows-per-file", type=int, default=200)
    ap.add_argument("--max-data-chars", type=int, default=120_000)
    ap.add_argument("--query-column", default="Query")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompts and report their size; no API calls")
    args = ap.parse_args()

    _check_env()

    cfg_raw = yaml.safe_load((HERE / "exp_config.yaml").read_text(encoding="utf-8"))
    test = load_dataset(args.dataset, cfg_raw)
    if args.rows:
        idx = [int(x) for x in args.rows.split(",") if x.strip() != ""]
        missing = [i for i in idx if i not in test.index]
        if missing:
            raise SystemExit(f"row indices not in the dataset: {missing}")
        test = test.loc[idx]
    if args.limit:
        test = test.head(args.limit)
    if args.query_column not in test.columns:
        raise SystemExit(f"no {args.query_column!r} column; "
                         f"have {list(test.columns)}")

    cfg = lx.load_config(HERE / "exp_config.yaml",
                         model_profile=args.profile,
                         method=args.method,
                         dataset=args.dataset,
                         repeat_index=args.repeat_index,
                         notes=f"single-call baseline, "
                               f"max_rows_per_file={args.max_rows_per_file}")

    if not args.dry_run:
        lx.ensure_api_keys(cfg)
    log = lx.RunLogger(cfg)

    # the baseline needs exactly one role; `modeler` is present in every profile
    llm = None if args.dry_run else lx.build_llm(cfg, "modeler")

    from langchain_core.messages import HumanMessage

    # Datasets disagree on column names (`Label-objective` vs `Label`, ...).
    # Resolve once, up front, and say what was found -- an unresolved
    # gold_objective is the difference between "scored" and "silently
    # unscoreable", so it should never be a surprise discovered later.
    colmap = lx.resolve_columns(cfg, test.columns)
    if args.query_column != "Query":
        colmap["query"] = args.query_column          # explicit flag wins
    if not colmap["query"]:
        raise SystemExit(f"no query column found; have {list(test.columns)}")
    print(f"[run_baseline] columns: {lx.describe_columns(colmap)}")
    if not colmap["gold_objective"]:
        print(f"[run_baseline] note: {args.dataset} ships no ground-truth "
              f"objective column, so score_runs.py cannot grade this run.")

    def cell(row, field):
        c = colmap.get(field)
        return row.get(c) if c else None

    rows_out = []
    for idx, row in test.iterrows():
        query = str(row[colmap["query"]])
        data_text, dstats = build_data_section(
            cell(row, "dataset_address"), args.max_rows_per_file,
            args.max_data_chars)
        prompt = PROMPT_TEMPLATE.format(query=query, data_section=data_text)
        approx = lx._estimate_tokens(prompt, cfg.token_fallback_encoder)

        if args.dry_run:
            print(f"[{idx}] prompt ~{approx:>7,} tokens | "
                  f"files={dstats['files']} "
                  f"rows {dstats['rows_included']}/{dstats['rows_total']}"
                  f"{' TRUNCATED' if dstats['truncated'] else ''}")
            rows_out.append({"instance_id": idx, "approx_prompt_tokens": approx,
                             **dstats})
            continue

        with log.instance(instance_id=int(idx), query=query,
                          gold_type=cell(row, "gold_type"),
                          dataset_address=cell(row, "dataset_address"),
                          size_class=cell(row, "size_class"),
                          gold_objective=cell(row, "gold_objective"),
                          prompt_chars=len(prompt),
                          **{f"data_{k}": v for k, v in dstats.items()}) as rec:
            answer, code = None, None
            try:
                with lx.stage("single_call"):
                    resp = lx.call_with_retry(
                        llm.invoke, cfg, [HumanMessage(content=prompt)])
                answer = resp.content
                code = extract_code(answer)
                rec.set(model_output=answer, code_output=code)
            except Exception as e:                    # noqa: BLE001
                print(f"[{idx}] failed: {type(e).__name__}: {e}")
                rec.fail(e)
                rec.set(model_output=None, code_output=None)

        s = log.records[-1].summary()
        print(f"[{idx}] {log.records[-1].status:7s} "
              f"in={s['prompt_tokens']:>7,} out={s['completion_tokens']:>6,} "
              f"${s['cost_usd']:.4f} {s['wall_s']:>5.1f}s "
              f"code={'yes' if code else 'NO'}"
              f"{' TRUNCATED-DATA' if dstats['truncated'] else ''}")
        rows_out.append({"instance_id": idx, "query": query,
                         "gold_type": cell(row, "gold_type"),
                         "model_output": answer, "code_output": code,
                         **{f"data_{k}": v for k, v in dstats.items()}})

    if args.dry_run:
        df = pd.DataFrame(rows_out)
        print(f"\n{len(df)} instances | "
              f"prompt tokens: median {df['approx_prompt_tokens'].median():,.0f}, "
              f"max {df['approx_prompt_tokens'].max():,.0f} | "
              f"{int(df['truncated'].sum())} truncated")
        print("no API calls made")
        return 0

    agg = log.close()
    out = log.dir_path / "baseline_predictions.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"\npredictions -> {out}")
    print(f"run dir     -> {log.dir_path}")
    print("\nnext:  python aggregate_runs.py runs/ -o tables/ --gold gold_labels.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
