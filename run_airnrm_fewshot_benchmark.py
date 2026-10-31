#!/usr/bin/env python3
"""Air-NRM one-shot benchmark using one exemplar from the matching LEANOPT code.

This runner keeps the existing direct single-call benchmark contract: each target
query causes exactly one model request, all raw Air-NRM CSV files are supplied in
that request, and the generated gurobipy program is scored against the query CSV's
gold objective.  The only experimental change is the addition of one worked
LEANOPT exemplar before the target query.
"""

from __future__ import annotations

import argparse
import ast
import getpass
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
AIR_ROOT = ROOT / "Test_Dataset" / "Air_NRM"
QUERY_FILES = {
    ("small", "ca"): (AIR_ROOT / "small_scale" / "query_CA.csv", "Air-NRM-CA"),
    ("small", "np"): (AIR_ROOT / "small_scale" / "query_NP_Flow.csv", "Air-NRM-NP"),
    ("large", "ca"): (AIR_ROOT / "large_scale" / "query_CA.csv", "Air-NRM-CA-LS"),
    ("large", "np"): (AIR_ROOT / "large_scale" / "query_NP_Flow.csv", "Air-NRM-NP-LS"),
}
DATA_FILES = {
    "small": [
        AIR_ROOT / "small_scale" / "flight.csv",
        AIR_ROOT / "small_scale" / "od_demand.csv",
        AIR_ROOT / "small_scale" / "v1.csv",
        AIR_ROOT / "small_scale" / "v2.csv",
    ],
    "large": [
        AIR_ROOT / "large_scale" / "flight.csv",
        AIR_ROOT / "large_scale" / "od_demand.csv",
        AIR_ROOT / "large_scale" / "v1.csv",
        AIR_ROOT / "large_scale" / "v2.csv",
        AIR_ROOT / "large_scale" / "flight_capacity.csv",
    ],
}
PROFILES = {
    "gpt-5.4": "gpt-5.4",
    "gemini-3.1-pro": "gemini-3.1-pro",
}
METHOD = "Base-SingleCall-LEANOPT-OneShot"

sys.path.insert(0, str(ROOT))
import leanopt_exp as lx  # noqa: E402


def load_core_module():
    path = ROOT / "Test_Dataset" / "Large-scale-or" / "run_test_universe_agentic.py"
    spec = importlib.util.spec_from_file_location("airnrm_one_shot_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark core from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_core_module()


def _verbatim_prompt(prompt: str, file_paths: list[str]) -> str:
    """Bypass the base runner's instructional file wrapper."""
    if file_paths:
        raise ValueError("verbatim one-shot requests must inline their raw data")
    return prompt


# The shared clients normally prepend guidance such as "Use only these files"
# and append a TASK heading.  This experiment explicitly forbids those hints.
CORE.prompt_with_file_contents = _verbatim_prompt


def parse_rows(value: str | None, n_rows: int) -> list[int]:
    if not value:
        return list(range(n_rows))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    bad = sorted(i for i in selected if i < 0 or i >= n_rows)
    if bad:
        raise ValueError(f"row index outside 0..{n_rows - 1}: {bad}")
    return sorted(selected)


def _safe_string_value(node: ast.AST, values: dict[str, str]) -> str:
    """Evaluate only string literals/f-strings used by the exemplar cells."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for item in node.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                parts.append(item.value)
            elif isinstance(item, ast.FormattedValue):
                parts.append(_safe_string_value(item.value, values))
            else:
                raise ValueError(f"unsupported f-string item: {ast.dump(item)}")
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_string_value(node.left, values) + _safe_string_value(node.right, values)
    raise ValueError(f"unsupported exemplar expression: {ast.dump(node)}")


def notebook_string_assignment(path: Path, target: str) -> str:
    """Read a string assignment without executing arbitrary notebook code."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    values: dict[str, str] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = node.targets[0]
            if not isinstance(name, ast.Name):
                continue
            try:
                values[name.id] = _safe_string_value(node.value, values)
            except ValueError:
                continue
            if name.id == target:
                return values[target]
    raise ValueError(f"string assignment {target!r} not found in {path}")


def small_np_exemplar() -> tuple[str, str]:
    from Test_Dataset.Air_NRM.small_scale import np_lean_workflow as workflow

    data = workflow.AirNrmSmallScaleData.from_path(AIR_ROOT / "small_scale")
    params = workflow.format_parameters(data.assemble(workflow.EXAMPLE_QUERY))
    exemplar = f'''Question: {workflow.EXAMPLE_QUERY}

Action: CName
Action Input: {workflow.EXAMPLE_QUERY}
Observation: identifiers preserved; call CSVQA next.

Action: CSVQA
Action Input: {workflow.EXAMPLE_QUERY}
Observation:
{params}
Final Answer:
### Objective Function
Maximize total expected ticket-sales revenue.

### Constraints
Decision variables; per-flight capacity; OD balance; scale; big-M
sales/selection linking; cardinality; airport flow conservation; domains.

### Generated Code
```python
{params}
{workflow.SMALL_EXAMPLE_MODEL_CODE}
```'''
    source = (
        "Test_Dataset/Air_NRM/small_scale/np_lean_workflow.py:"
        "EXAMPLE_QUERY+SMALL_EXAMPLE_MODEL_CODE"
    )
    return exemplar, source


def large_np_exemplar() -> tuple[str, str]:
    from Test_Dataset.Air_NRM.large_scale import np_lean_workflow as workflow

    data = workflow.AirNrmLargeScaleData.from_path(AIR_ROOT / "large_scale")
    example_path = AIR_ROOT / "large_scale" / "RAG_Example_SBLP_Flow_LargeScale_NP.csv"
    query = workflow._read_example_query(example_path)
    params = workflow.format_parameters(data.assemble(query))
    cname = (
        query
        + "\n\nCNAME_STATUS: identifiers_only; numeric_model_parameters_not_retrieved."
          "\nMANDATORY_NEXT_ACTION: call CSVQA with the complete query above before "
          "writing any Final Answer."
    )
    exemplar = f'''Question: {query}

Thought: I must preserve the complete candidate-flight list, Z, capacity percentage,
and all four capacity-consumption coefficients.
Action: CName
Action Input: {query}
Observation: {cname}

Thought: I need the exact model parameters from the five Air-NRM CSV files.
Action: CSVQA
Action Input: {query}
Observation:
{params}
Thought: I must copy every assignment from the CSVQA observation exactly once,
then append the complete NP Gurobi model in the same code block.

Final Answer:
### Objective Function
Maximize total expected ticket-sales revenue:

`maximize sum(avg_price[fid,pid] * x[fid,pid])`.

### Constraints

#### 1. Decision Variables
`x[fid,pid] >= 0` is sales of product `pid` on flight `fid`; `x_o[od] >= 0`
is no-purchase demand; `y[fid]` is binary flight selection.

#### 2. Capacity Constraints
For each flight, capacity-weighted sales cannot exceed `flight_capacity[fid]`.

#### 3. Balance Constraints
For each OD, ratio-weighted no-purchase demand plus ratio-weighted ticket sales
equals `avg_pax[od]`.

#### 4. Scale Constraints
For each ticket option, `value_0_list[od] * x[fid,pid]` cannot exceed
`value_list[fid,pid] * x_o[od]`.

#### 5. Sales/Selection Linking
Sales are zero unless the flight is selected, using the tight capacity-derived
upper bound `flight_capacity[fid] / capacity_consum[pid]`.

#### 6. Cardinality Constraint
At most `option_num` flights may be selected.

#### 7. Flow Conservation Constraints
At every airport, the number of selected inbound flights equals the number of
selected outbound flights.

#### 8. Variable Domains
Sales and no-purchase variables are nonnegative; flight selections are binary.

### Generated Code
```python
{params}
{workflow.EXAMPLE_MODEL_CODE}
```'''
    source = (
        "Test_Dataset/Air_NRM/large_scale/np_lean_workflow.py:"
        "make_fewshot+EXAMPLE_MODEL_CODE; "
        "Test_Dataset/Air_NRM/large_scale/RAG_Example_SBLP_Flow_LargeScale_NP.csv"
    )
    return exemplar, source


def get_exemplar(scale: str, family: str) -> tuple[str, str]:
    if family == "np":
        return small_np_exemplar() if scale == "small" else large_np_exemplar()

    if scale == "small":
        path = ROOT / "Ablation_Study_Air_NRM_Few-shot_Only.ipynb"
        value = notebook_string_assignment(path, "fewshot_only_CA")
        source = f"{path.relative_to(ROOT)}:fewshot_only_CA"
        # In the notebook this string goes through LangChain PromptTemplate,
        # which turns doubled literal braces back into normal dictionary
        # braces.  This runner calls the provider directly, so perform that
        # same final rendering step here.  Also remove the historical single
        # leading space on top-level parameter assignments so the shown Python
        # block is syntactically executable.
        value = value.replace("{{", "{").replace("}}", "}")
        value = re.sub(
            r"(?m)^ +(?=(?:value_list|ratio_list|value_0_list|ratio_0_list|"
            r"avg_pax|capacity_consum|flight_capacity)\s*=)",
            "",
            value,
        )
        source += " [LangChain brace rendering normalized by runner]"
    else:
        path = AIR_ROOT / "large_scale" / "Ablation_Study_Air_NRM_Few-shot_Only_LargeScale.ipynb"
        value = notebook_string_assignment(path, "FEWSHOT_CA")
        source = f"{path.relative_to(ROOT)}:FEWSHOT_CA"
    value = value.strip()
    # The two historical CA notebook strings open a Python fence but omit the
    # closing delimiter.  Close only that transport delimiter so the target
    # query cannot accidentally become part of the demonstration code block;
    # the exemplar text and code themselves remain unchanged.
    if value.count("```") % 2:
        note = "\nNote: Please retrieve"
        if note in value:
            value = value.replace(note, "\n```\n\nNote: Please retrieve", 1)
        else:
            value += "\n```"
        source += " [closing fence normalized by runner]"
    return value, source


def build_prompt(query: str, exemplar: str, data_files: list[Path]) -> str:
    """Return exactly one exemplar, raw benchmark data, and the target query.

    There are deliberately no role instructions, response-format requirements,
    reminders, or task hints.  Filenames merely delimit the raw CSV bytes that
    are part of the benchmark instance rather than adding natural-language help.
    """
    parts = [exemplar]
    for path in data_files:
        contents = path.read_text(encoding="utf-8-sig")
        parts.append(f"{path.name}\n{contents}")
    parts.append(f"Question: {query}")
    return "\n\n".join(parts)


def call_cost(metadata: dict, model_spec: lx.ModelSpec) -> dict:
    prompt = int(metadata.get("prompt_tokens") or 0)
    cached = min(prompt, int(metadata.get("cached_prompt_tokens") or 0))
    output = int(metadata.get("completion_tokens") or 0)
    if model_spec.provider == "google":
        output += int(metadata.get("reasoning_tokens") or 0)
    extra = model_spec.extra or {}
    if prompt > int(extra.get("long_context_threshold", 10**18)):
        price_in = float(extra.get("long_context_price_in_per_1m", model_spec.price_in_per_1m))
        price_cached = float(extra.get("long_context_price_cached_in_per_1m", model_spec.price_cached_in_per_1m))
        price_out = float(extra.get("long_context_price_out_per_1m", model_spec.price_out_per_1m))
        tier = "long_context"
    else:
        price_in = model_spec.price_in_per_1m
        price_cached = model_spec.price_cached_in_per_1m
        price_out = model_spec.price_out_per_1m
        tier = "standard"
    cost = ((prompt - cached) * price_in + cached * price_cached + output * price_out) / 1_000_000
    return {
        "cost_usd": round(cost, 8),
        "token_cost_usd": round(cost, 8),
        "tool_cost_usd": 0.0,
        "price_tier": tier,
        "price_in_per_1m": price_in,
        "price_cached_in_per_1m": price_cached,
        "price_out_per_1m": price_out,
    }


def prompt_for_api_key(spec: lx.ModelSpec) -> None:
    provider = "Gemini" if spec.provider == "google" else "OpenAI"
    key = getpass.getpass(f"Enter {provider} API key (input hidden): ").strip()
    if len(key) < 20:
        raise ValueError(f"{provider} API key looks empty or incomplete")
    os.environ[spec.api_key_env] = key
    print(f"{provider} API key accepted: ****{key[-4:]}")


def build_client_config(cfg: lx.ExpConfig, args) -> dict:
    spec = cfg.spec("modeler")
    return {
        "model_type": "gemini" if spec.provider == "google" else "openai",
        "model_name": spec.model,
        "gemini_api_key": os.getenv(spec.api_key_env, ""),
        "openai_api_key": os.getenv(spec.api_key_env, ""),
        "max_retries": 1,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": spec.max_tokens,
        "if_proxy": args.if_proxy,
        "proxy_url": args.proxy_url,
    }


def logged_request(client, spec: lx.ModelSpec, rec, prompt: str,
                   data_files: list[Path]) -> str:
    started = time.perf_counter()
    try:
        metadata = client.generate_with_metadata(
            prompt,
            [],
            system_instructions="",
        )
        text = metadata.pop("text", "")
        rec.add_call({
            "type": "llm",
            "stage": "single_call_one_shot",
            "ok": True,
            "provider": spec.provider,
            "model": metadata.pop("model", spec.model),
            "prompts": [prompt],
            "response": text,
            **metadata,
            **call_cost(metadata, spec),
        })
        return text
    except Exception as exc:
        rec.add_call({
            "type": "llm",
            "stage": "single_call_one_shot",
            "ok": False,
            "provider": spec.provider,
            "model": spec.model,
            "prompts": [prompt],
            "response": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "latency_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def generate_report(run_dir: Path, timeout: float) -> dict:
    report_path = AIR_ROOT / "large_scale" / "report_agentic_benchmark.py"
    spec = importlib.util.spec_from_file_location("airnrm_one_shot_report", report_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load report generator from {report_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_report(run_dir, timeout=timeout, diagnose=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scale", choices=("small", "large"))
    ap.add_argument("family", choices=("ca", "np"))
    ap.add_argument("profile", choices=tuple(PROFILES))
    ap.add_argument("--rows", help="zero-based rows, e.g. 0,2-4")
    ap.add_argument("--limit", type=int, help="run only the first N selected rows")
    ap.add_argument("--repeat-index", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--max-run-cost-usd", type=float, default=45.0)
    ap.add_argument("--prompt-key", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    proxy = ap.add_mutually_exclusive_group()
    proxy.add_argument("--if-proxy", dest="if_proxy", action="store_true")
    proxy.add_argument("--no-proxy", dest="if_proxy", action="store_false")
    ap.set_defaults(if_proxy=False)
    ap.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    args = ap.parse_args()
    if args.max_run_cost_usd < 0:
        ap.error("--max-run-cost-usd must be nonnegative")

    query_csv, dataset_name = QUERY_FILES[(args.scale, args.family)]
    data_files = DATA_FILES[args.scale]
    missing = [str(path) for path in [query_csv, *data_files] if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing benchmark file(s): " + ", ".join(missing))

    data = pd.read_csv(query_csv)
    required = {"Query", "Label-objective"}
    if not required.issubset(data.columns):
        raise ValueError(f"{query_csv} must contain {sorted(required)}")
    rows = parse_rows(args.rows, len(data))
    if args.limit is not None:
        rows = rows[:args.limit]

    exemplar, exemplar_source = get_exemplar(args.scale, args.family)
    exemplar_sha = hashlib.sha256(exemplar.encode("utf-8")).hexdigest()
    overlap_rows = [
        idx for idx, query in enumerate(data["Query"].astype(str))
        if query.strip() and query.strip() in exemplar
    ]
    if overlap_rows:
        raise ValueError(
            "LEANOPT exemplar duplicates target query row(s), which would leak the "
            f"answer into the benchmark: {overlap_rows}"
        )
    cfg = lx.load_config(
        ROOT / "exp_config.yaml",
        model_profile=PROFILES[args.profile],
        method=METHOD,
        dataset=dataset_name,
        repeat_index=args.repeat_index,
        solver_time_limit_s=args.timeout,
        budget_usd_per_run=args.max_run_cost_usd,
    )
    cfg.manifest["one_shot_policy"] = {
        "model_requests_per_query": 1,
        "leanopt_exemplars_per_query": 1,
        "exemplar_source": exemplar_source,
        "exemplar_sha256": exemplar_sha,
        "exemplar_duplicates_target_query": False,
        "target_query_inserted_verbatim": True,
        "data_delivery": "complete_inline_csv_text",
        "system_instructions": "",
        "extra_task_instructions": False,
        "payload_order": ["one_leanopt_exemplar", "raw_csv_data", "target_query"],
        "accuracy_tolerance": "relative_1e-4",
    }

    sample_prompt = (
        build_prompt(str(data.iloc[rows[0]]["Query"]), exemplar, data_files)
        if rows else ""
    )
    print(f"scale={args.scale} family={args.family} dataset={dataset_name}")
    print(f"profile={args.profile} model={cfg.spec('modeler').model}")
    print(f"instances={rows}")
    print(f"exemplar={exemplar_source}")
    print(f"exemplar_sha256={exemplar_sha}")
    print(f"sample_prompt_chars={len(sample_prompt):,}; data_files={len(data_files)}")
    print(f"cost_limit_per_run_usd={args.max_run_cost_usd:.2f} (fresh counter for this run)")
    if args.dry_run:
        print("dry-run passed; no API call or run directory was created")
        return 0

    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if args.prompt_key:
        prompt_for_api_key(cfg.spec("modeler"))
    else:
        lx.ensure_api_keys(cfg, interactive=False)

    CORE.setup_proxy({"if_proxy": args.if_proxy, "proxy_url": args.proxy_url})
    client = CORE.create_client(build_client_config(cfg, args))
    logger = lx.RunLogger(cfg)
    spec = cfg.spec("modeler")
    relative_addresses = "\n".join(str(path.relative_to(ROOT)) for path in data_files)
    budget_stop = ""
    try:
        try:
            for iid in rows:
                if args.max_run_cost_usd and logger.spent_usd() >= args.max_run_cost_usd:
                    budget_stop = (
                        f"run cost ${logger.spent_usd():.4f} reached "
                        f"${args.max_run_cost_usd:.2f}"
                    )
                    break
                row = data.iloc[iid]
                query = str(row["Query"])
                prompt = build_prompt(query, exemplar, data_files)
                with logger.instance(
                    iid,
                    query=query,
                    gold_objective=float(row["Label-objective"]),
                    dataset_address=relative_addresses,
                    size_class=row.get("n_variables", row.get("Decision Variables")),
                    exemplar_source=exemplar_source,
                    exemplar_sha256=exemplar_sha,
                ) as rec:
                    try:
                        response = logged_request(client, spec, rec, prompt, data_files)
                        math_model, code, extraction_error = CORE.extract_code_from_response(response)
                        if code and not extraction_error:
                            code = CORE.replace_file_paths_in_code(
                                code, [str(path.resolve()) for path in data_files]
                            )
                        rec.set(
                            model_output=response,
                            math_model=math_model,
                            code_output=code,
                            extraction_error=extraction_error,
                        )
                        if extraction_error:
                            rec.fail(RuntimeError(extraction_error), "response extraction")
                    except Exception as exc:
                        rec.fail(exc, "provider request")
                instance_summary = logger.records[-1].summary()
                print(
                    f"[{iid}] {logger.records[-1].status} "
                    f"in={instance_summary['prompt_tokens']} "
                    f"out={instance_summary['completion_tokens']} "
                    f"${instance_summary['cost_usd']:.4f}"
                )
        except lx.BudgetExceeded as exc:
            budget_stop = str(exc)
    finally:
        summary = logger.close()

    if budget_stop:
        summary["safety_stop"] = budget_stop
        (logger.dir_path / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"SAFETY STOP: {budget_stop}")

    report = generate_report(logger.dir_path, args.timeout)
    score = report["scores"]["relative_tolerance_1e-4"]
    print(
        f"accuracy@1e-4: {score['correct']}/{score['scored']} "
        f"({100 * score['accuracy']:.2f}%)"
        if score["accuracy"] is not None
        else "accuracy@1e-4: not scored"
    )
    print(f"report: {(logger.dir_path / 'report.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
