#!/usr/bin/env python3
"""Logged Air-NRM inline-data benchmarks for Gemini/GPT-5.4."""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QUERY_FILES = {
    "ca": (HERE / "query_CA.csv", "Air-NRM-CA-LS"),
    "np": (HERE / "query_NP_Flow.csv", "Air-NRM-NP-LS"),
}
# Backward-compatible name used by the existing offline regression tests.
QUERY_CSV = QUERY_FILES["ca"][0]
DATA_FILES = [
    HERE / "flight.csv",
    HERE / "od_demand.csv",
    HERE / "v1.csv",
    HERE / "v2.csv",
    HERE / "flight_capacity.csv",
]
PROFILES = {
    "gemini": "gemini-3.1-pro",
    "gemini-3.1-pro": "gemini-3.1-pro",
    "openai": "gpt-5.4",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.2": "gpt-5.2",
    "gemini-3-pro": "gemini-3-pro",
}

TWO_STAGE_METHOD = "Base-Agentic-2Stage-1Repair"
FORMULATION_SYSTEM_INSTRUCTIONS = (
    "You are an expert in mathematical programming, operations research, and "
    "mathematical optimization. Produce the mathematical formulation requested "
    "by the user. Do not write Python or other implementation code in this stage."
)
CODE_SYSTEM_INSTRUCTIONS = (
    "You are an expert in operations research and gurobipy. Translate the supplied "
    "mathematical formulation into one complete, directly executable Python program "
    "using gurobipy. Return exactly one fenced ```python ... ``` code block and no "
    "other code block. The program must import gurobipy, create a gurobipy Model, "
    "call optimize(), and leave Gurobi output enabled so the optimal objective is "
    "visible. Use only the five supplied CSV filenames."
)
REPAIR_SYSTEM_INSTRUCTIONS = (
    "You are repairing a gurobipy program after one local execution attempt. Use the "
    "supplied formulation, previous program, and local execution diagnostics. Return "
    "exactly one complete fenced ```python ... ``` replacement program and no other "
    "code block. Do not request another execution or explain the repair."
)

sys.path.insert(0, str(ROOT))
import leanopt_exp as lx  # noqa: E402


def load_agentic_module():
    path = ROOT / "Test_Dataset" / "Large-scale-or" / "run_test_universe_agentic.py"
    spec = importlib.util.spec_from_file_location("airnrm_agentic_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load agentic core from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_agentic_module()


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


def call_cost(metadata: dict, model_spec: lx.ModelSpec) -> dict:
    prompt = int(metadata.get("prompt_tokens") or 0)
    cached = min(prompt, int(metadata.get("cached_prompt_tokens") or 0))
    output = int(metadata.get("completion_tokens") or 0)
    # Gemini reports visible candidate tokens and hidden thinking tokens in
    # separate fields, while OpenAI's output_tokens already includes its
    # reasoning-token subset. Both are billable output for Gemini only.
    if model_spec.provider == "google":
        output += int(metadata.get("reasoning_tokens") or 0)
    extra = model_spec.extra or {}

    if prompt > int(extra.get("long_context_threshold", 10**18)):
        price_in = float(extra.get("long_context_price_in_per_1m",
                                   model_spec.price_in_per_1m))
        price_cached = float(extra.get("long_context_price_cached_in_per_1m",
                                       model_spec.price_cached_in_per_1m))
        price_out = float(extra.get("long_context_price_out_per_1m",
                                    model_spec.price_out_per_1m))
        price_tier = "long_context"
    else:
        price_in = model_spec.price_in_per_1m
        price_cached = model_spec.price_cached_in_per_1m
        price_out = model_spec.price_out_per_1m
        price_tier = "standard"

    token_cost = ((prompt - cached) * price_in + cached * price_cached
                  + output * price_out) / 1_000_000
    tool_cost = float(extra.get("container_cost_usd", 0.0))
    return {
        "cost_usd": round(token_cost + tool_cost, 8),
        "token_cost_usd": round(token_cost, 8),
        "tool_cost_usd": round(tool_cost, 8),
        "price_tier": price_tier,
        "price_in_per_1m": price_in,
        "price_cached_in_per_1m": price_cached,
        "price_out_per_1m": price_out,
    }


def build_client_config(profile: str, cfg: lx.ExpConfig, args) -> dict:
    spec = cfg.spec("modeler")
    return {
        "model_type": "gemini" if spec.provider == "google" else "openai",
        "model_name": spec.model,
        "gemini_api_key": os.getenv(spec.api_key_env, ""),
        "openai_api_key": os.getenv(spec.api_key_env, ""),
        "max_retries": cfg.max_retries,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "max_tokens": spec.max_tokens,
        "if_proxy": args.if_proxy,
        "proxy_url": args.proxy_url,
    }


def prompt_for_api_key(model_spec: lx.ModelSpec) -> None:
    """Read a provider key without terminal echo and keep it process-local."""
    provider_name = "Gemini" if model_spec.provider == "google" else "OpenAI"
    key = getpass.getpass(f"Enter {provider_name} API key (input hidden): ").strip()
    if len(key) < 20:
        raise ValueError(f"{provider_name} API key looks empty or incomplete")
    os.environ[model_spec.api_key_env] = key
    print(f"{provider_name} API key accepted: ****{key[-4:]}")


def instance_cost_limit(row, all_rows: pd.DataFrame, args) -> float:
    """Allocate 25% equally for fixed overhead and 75% by model size."""
    if args.max_instance_cost_usd is not None:
        return args.max_instance_cost_usd
    if not args.max_run_cost_usd:
        return 0.0
    n_instances = len(all_rows)
    size_column = next((name for name in ("n_variables", "Decision Variables")
                        if name in all_rows.columns), None)
    if size_column is None:
        return args.max_run_cost_usd / n_instances
    total_variables = float(all_rows[size_column].sum())
    fixed_share = 0.25 * args.max_run_cost_usd / n_instances
    variable_share = (
        0.75 * args.max_run_cost_usd
        * float(row[size_column]) / total_variables
    )
    return fixed_share + variable_share


def run_report(run_dir: Path, timeout: float) -> None:
    from report_agentic_benchmark import generate_report
    generate_report(run_dir, timeout=timeout)


def build_formulation_prompt(query: str) -> str:
    return f"""Below is an operations research question.

# Question

{query}

# Stage 1

Write the mathematical formulation only. Define the sets, parameters, decision
variables, objective, and constraints precisely, including integrality when the
question requires it. Do not generate Python code yet.
"""


def build_code_prompt(query: str, formulation: str) -> str:
    return f"""Below is the original benchmark question, followed by the mathematical
formulation produced in Stage 1.

# Original Question

{query}

# Stage 1 Mathematical Formulation

{formulation}

# Stage 2

Generate one complete, directly executable gurobipy Python program implementing the
formulation. Return exactly one fenced Python code block.
"""


def build_repair_prompt(query: str, formulation: str, code: str,
                        execution: dict) -> str:
    diagnostics = json.dumps(execution, indent=2, default=str)
    return f"""The first gurobipy program did not complete with a parseable optimal
objective. Repair it once.

# Original Question

{query}

# Mathematical Formulation

{formulation}

# Previous Python Program

```python
{code}
```

# Local Execution Diagnostics

```text
{diagnostics}
```

Return one complete replacement gurobipy program in exactly one fenced Python code
block. This is the final repair attempt.
"""


def logged_model_request(client, spec: lx.ModelSpec, rec, stage: str,
                         prompt: str, system_instructions: str) -> tuple[str, dict]:
    """Make and fully log exactly one provider request; never retry here."""
    started = time.perf_counter()
    try:
        metadata = client.generate_with_metadata(
            prompt,
            [str(p) for p in DATA_FILES],
            system_instructions=system_instructions,
        )
        raw_text = metadata.pop("text", "")
        prices = call_cost(metadata, spec)
        rec.add_call({
            "type": "llm",
            "stage": stage,
            "ok": True,
            "provider": spec.provider,
            "model": metadata.pop("model", spec.model),
            "prompts": [prompt],
            "response": raw_text,
            **metadata,
            **prices,
        })
        return raw_text, prices
    except Exception as exc:
        rec.add_call({
            "type": "llm", "stage": stage, "ok": False,
            "provider": spec.provider, "model": spec.model,
            "prompts": [prompt], "response": "",
            "prompt_tokens": 0, "completion_tokens": 0,
            "cached_prompt_tokens": 0, "reasoning_tokens": 0,
            "total_tokens": 0, "cost_usd": 0.0,
            "token_source": "unavailable",
            "latency_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def execute_local_attempt(rec, stage: str, code: str,
                          extraction_error: str, timeout: float) -> tuple[dict, bool]:
    """Run at most one local process and log it as a neutral benchmark tool."""
    started = time.perf_counter()
    if extraction_error:
        result = {
            "ok": False,
            "returncode": None,
            "objective": None,
            "stdout": "",
            "stderr": "",
            "error": f"code validation failed: {extraction_error}",
        }
    else:
        stdout, stderr, returncode = CORE.run_generated_code(code, timeout)
        objective = CORE.extract_obj_from_gurobi_log(stdout + "\n" + stderr)
        result = {
            "ok": returncode == 0 and objective is not None,
            "returncode": returncode,
            "objective": objective,
            "stdout": stdout,
            "stderr": stderr,
            "error": "",
        }
        if returncode != 0:
            result["error"] = f"local Python process exited with code {returncode}"
        elif objective is None:
            result["error"] = "no parseable optimal objective in local execution output"
    latency = round(time.perf_counter() - started, 3)
    rec.add_call({
        "type": "tool",
        "stage": stage,
        "tool_name": "local_gurobipy_execution",
        "ok": result["ok"],
        "latency_s": latency,
        "input_code_chars": len(code or ""),
        "returncode": result["returncode"],
        "objective": result["objective"],
        "error": result["error"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    })
    result["latency_s"] = latency
    return result, bool(result["ok"])


def main() -> int:
    global DATA_FILES, CODE_SYSTEM_INSTRUCTIONS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profile", choices=sorted(PROFILES),
                    help="model profile from exp_config.yaml")
    ap.add_argument("--scale", choices=("medium", "large"), default="large")
    ap.add_argument(
        "--dataset",
        choices=tuple(QUERY_FILES),
        default="ca",
        help="Air-NRM large-scale query family: ca (default) or np",
    )
    ap.add_argument(
        "--workflow",
        choices=("single", "two-stage-repair"),
        default="single",
        help=("single: existing one-request baseline; two-stage-repair: "
              "formulation, code, and at most one repair"),
    )
    ap.add_argument("--rows", help="zero-based rows, e.g. 0,2-4")
    ap.add_argument("--limit", type=int, help="run only the first N selected rows")
    ap.add_argument("--repeat-index", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="per-instance solver timeout used by the final report")
    ap.add_argument("--max-instance-cost-usd", type=float, default=None,
                    help="fixed per-instance limit; default dynamically allocates the run budget by n_variables")
    ap.add_argument("--max-run-cost-usd", type=float, default=30.0,
                    help="stop before another instance after total cost reaches this (default: 30)")
    proxy = ap.add_mutually_exclusive_group()
    proxy.add_argument("--if-proxy", "--if_proxy", dest="if_proxy",
                       action="store_true", help="enable proxy explicitly")
    proxy.add_argument("--no-proxy", "--no_proxy", dest="if_proxy",
                       action="store_false", help="use a direct connection (default)")
    ap.set_defaults(if_proxy=False)
    ap.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--prompt-key", action="store_true",
                    help="securely prompt for this run's API key and show only its last 4 characters")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate data/config without making API calls")
    args = ap.parse_args()
    if ((args.max_instance_cost_usd is not None
         and args.max_instance_cost_usd < 0)
            or args.max_run_cost_usd < 0):
        ap.error("cost limits must be nonnegative (0 disables that limit)")

    profile = PROFILES[args.profile]
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    method = "Base-SingleCall" if args.workflow == "single" else TWO_STAGE_METHOD
    query_csv, dataset_name = QUERY_FILES[args.dataset]
    if args.scale == "medium":
        medium = HERE.parent / "medium_scale"
        query_csv = medium / ("query_CA.csv" if args.dataset == "ca"
                              else "query_NP_Flow.csv")
        dataset_name = "Air-NRM-CA" if args.dataset == "ca" else "Air-NRM-NP"
        DATA_FILES = [medium / name for name in
                      ("flight.csv", "od_demand.csv", "v1.csv", "v2.csv",
                       "flight_capacity.csv")]
    cfg = lx.load_config(
        ROOT / "exp_config.yaml",
        model_profile=profile,
        method=method,
        dataset=dataset_name,
        repeat_index=args.repeat_index,
        solver_time_limit_s=args.timeout,
        budget_usd_per_run=args.max_run_cost_usd,
    )
    # The medium directory was renamed from small_scale; fingerprint the files
    # actually supplied, including raw CSVs, rather than the stale YAML path.
    cfg.dataset_path = str(query_csv.relative_to(ROOT))
    cfg.manifest["dataset"] = lx.dataset_fingerprint(str(query_csv))
    cfg.manifest["inline_data_files"] = [lx.dataset_fingerprint(str(p))
                                         for p in DATA_FILES]
    cfg.manifest["benchmark_scale"] = args.scale
    cfg.manifest["instance_cost_size_column"] = (
        "n_variables" if args.scale == "large" else
        "Decision Variables" if args.dataset == "np" else None
    )
    if args.scale == "medium":
        cfg.manifest["benchmark_adapter"] = "medium_resources_v2_capacity_input"
        cfg.notes = (cfg.notes + " medium_resources_v2_capacity_input").strip()
        # These baselines use no embeddings, even for a profile whose defaults
        # include retrieval roles. Require only the model provider's API key.
        cfg.embedding_provider = "none"
        stem = cfg.run_id.rsplit("__", 1)[0]
        cfg.run_id = f"{stem}__{cfg.config_hash()}"
    cfg.manifest["agentic_safety"] = {
        "max_instance_cost_usd": args.max_instance_cost_usd,
        "max_run_cost_usd": args.max_run_cost_usd,
        "instance_cost_policy": (
            "fixed_override" if args.max_instance_cost_usd is not None
            else "25pct_equal_overhead_plus_75pct_n_variables"
        ),
        "enforcement": "post_response",
    }
    if args.workflow == "single":
        cfg.manifest["request_policy"] = {
            "workflow": "single",
            "model_requests_per_query": 1,
            "local_execution_attempts_during_generation": 0,
            "sdk_retries": 0,
            "hosted_execution_tools": False,
            "data_delivery": "complete_inline_csv_text",
            "preserve_output_when_code_invalid": True,
        }
    else:
        cfg.manifest["request_policy"] = {
            "workflow": "formulation_then_code_then_optional_single_repair",
            "model_requests_per_query_min": 2,
            "model_requests_per_query_max": 3,
            "local_execution_attempts_min": 1,
            "local_execution_attempts_max": 2,
            "repair_condition": "local execution lacks a parseable optimal objective",
            "repair_attempts_max": 1,
            "sdk_retries": 0,
            "hosted_execution_tools": False,
            "execution_tool": "local_gurobipy_execution",
            "uses_leanopt_model_or_workflow": False,
            "gold_objective_visible_to_model": False,
            "data_delivery": "complete_inline_csv_text_each_stage",
            "preserve_all_stage_outputs": True,
        }
        cfg.manifest["workflow_prompts"] = {
            "formulation_system": FORMULATION_SYSTEM_INSTRUCTIONS,
            "code_system": CODE_SYSTEM_INSTRUCTIONS,
            "repair_system": REPAIR_SYSTEM_INSTRUCTIONS,
            "original_query_is_inserted_verbatim": True,
        }
    CORE.setup_proxy({"if_proxy": args.if_proxy, "proxy_url": args.proxy_url})

    missing = [str(p) for p in [query_csv, *DATA_FILES] if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing benchmark file(s): " + ", ".join(missing))
    if args.scale == "medium":
        capacity = pd.read_csv(medium / "flight_capacity.csv")
        flight_keys = pd.read_csv(medium / "flight.csv")[["Oneway_OD", "Departure Time"]].drop_duplicates()
        capacity_keys = capacity[["Oneway_OD", "Departure Time"]]
        if (capacity_keys.duplicated().any()
                or set(map(tuple, capacity_keys.to_numpy())) != set(map(tuple, flight_keys.to_numpy()))
                or not pd.to_numeric(capacity["Capacity"], errors="coerce").gt(0).all()):
            raise ValueError("medium flight_capacity.csv must supply one positive Capacity for every flight")
    data = pd.read_csv(query_csv)
    required = {"Query", "Label-objective"}
    if not required.issubset(data.columns):
        raise ValueError(f"{query_csv} must contain {sorted(required)}")
    rows = parse_rows(args.rows, len(data))
    if args.limit is not None:
        rows = rows[:args.limit]

    print(f"profile={profile}  model={cfg.spec('modeler').model}")
    print(f"workflow={args.workflow}  method={method}")
    print(f"scale={args.scale} dataset={dataset_name}  query_file={query_csv.relative_to(ROOT)}")
    print(f"instances={rows}  data_files={len(DATA_FILES)}")
    if args.max_instance_cost_usd is None:
        limits = [instance_cost_limit(row, data, args)
                  for _, row in data.iterrows()]
        print("cost limits: allocated by available variable counts (equal if absent) "
              f"(${min(limits):.2f}..${max(limits):.2f}/instance), "
              f"${args.max_run_cost_usd:.2f}/run (checked after each response)")
    else:
        print("cost limits: "
              f"${args.max_instance_cost_usd:.2f}/instance, "
              f"${args.max_run_cost_usd:.2f}/run (checked after each response)")
    if args.dry_run:
        print("dry-run passed; no API call or run directory was created")
        return 0

    if args.prompt_key:
        prompt_for_api_key(cfg.spec("modeler"))
    else:
        lx.ensure_api_keys(cfg, interactive=False)
    client_config = build_client_config(profile, cfg, args)
    client = CORE.create_client(client_config)
    logger = lx.RunLogger(cfg)
    relative_addresses = "\n".join(str(p.relative_to(ROOT)) for p in DATA_FILES)
    spec = cfg.spec("modeler")
    safety_stop = ""

    try:
        for iid in rows:
            if (args.max_run_cost_usd
                    and logger.spent_usd() >= args.max_run_cost_usd):
                safety_stop = (
                    f"run cost ${logger.spent_usd():.4f} reached the "
                    f"${args.max_run_cost_usd:.2f} limit"
                )
                break
            row = data.iloc[iid]
            instance_cap = instance_cost_limit(row, data, args)
            query = str(row["Query"])
            gold = float(row["Label-objective"])
            instance_cost_stop = ""
            try:
                with logger.instance(
                    iid,
                    query=query,
                    gold_objective=gold,
                    dataset_address=relative_addresses,
                    size_class=row.get("n_variables", row.get("Decision Variables")),
                    cost_limit_usd=round(instance_cap, 6),
                ) as rec:
                    if args.workflow == "single":
                        prompt = CORE.build_prompt(query)
                        try:
                            raw_text, prices = logged_model_request(
                                client, spec, rec, "modeler", prompt,
                                CORE.SYSTEM_INSTRUCTIONS,
                            )
                        except Exception as exc:
                            rec.fail(exc, "provider request")
                            continue

                        math_model, code, extraction_error = (
                            CORE.extract_code_from_response(raw_text)
                        )
                        if code and not extraction_error:
                            code = CORE.replace_file_paths_in_code(
                                code, [str(p) for p in DATA_FILES]
                            )
                        rec.set(
                            model_output=raw_text,
                            math_model=math_model,
                            code_output=code,
                            extraction_error=extraction_error,
                        )
                        if extraction_error:
                            rec.fail(
                                RuntimeError(extraction_error),
                                "response extraction",
                            )
                    else:
                        print(f"[{iid}] Stage 1/3: generating LP formulation...",
                              flush=True)
                        try:
                            formulation, _ = logged_model_request(
                                client, spec, rec, "formulation",
                                build_formulation_prompt(query),
                                FORMULATION_SYSTEM_INSTRUCTIONS,
                            )
                        except Exception as exc:
                            rec.fail(exc, "formulation request")
                            continue
                        rec.set(formulation_output=formulation,
                                math_model=formulation)

                        current_instance_cost = rec.summary()["cost_usd"]
                        projected_run_cost = logger.spent_usd() + current_instance_cost
                        if (instance_cap
                                and current_instance_cost > instance_cap):
                            instance_cost_stop = (
                                f"instance {iid} cost ${current_instance_cost:.4f}, "
                                f"over its ${instance_cap:.2f} limit after formulation"
                            )
                            rec.fail(RuntimeError(instance_cost_stop),
                                     "cost safety stop")
                            continue
                        if (args.max_run_cost_usd
                                and projected_run_cost > args.max_run_cost_usd):
                            safety_stop = (
                                f"run cost ${projected_run_cost:.4f} exceeded the "
                                f"${args.max_run_cost_usd:.2f} limit after formulation"
                            )
                            rec.fail(RuntimeError(safety_stop), "run cost safety stop")
                            continue

                        print(f"[{iid}] Stage 2/3: generating gurobipy code...",
                              flush=True)
                        try:
                            code_response, _ = logged_model_request(
                                client, spec, rec, "code_generation",
                                build_code_prompt(query, formulation),
                                CODE_SYSTEM_INSTRUCTIONS,
                            )
                        except Exception as exc:
                            rec.fail(exc, "code generation request")
                            continue
                        _, initial_code, initial_extraction_error = (
                            CORE.extract_code_from_response(code_response)
                        )
                        if initial_code and not initial_extraction_error:
                            initial_code = CORE.replace_file_paths_in_code(
                                initial_code, [str(p) for p in DATA_FILES]
                            )
                        rec.set(
                            code_generation_output=code_response,
                            initial_code_output=initial_code,
                            initial_extraction_error=initial_extraction_error,
                        )

                        print(f"[{iid}] Local execution 1/2...", flush=True)
                        initial_execution, initial_ok = execute_local_attempt(
                            rec, "initial_local_execution", initial_code,
                            initial_extraction_error, args.timeout,
                        )
                        rec.set(initial_execution=initial_execution)

                        current_instance_cost = rec.summary()["cost_usd"]
                        projected_run_cost = logger.spent_usd() + current_instance_cost
                        over_instance_cap = bool(
                            instance_cap and current_instance_cost > instance_cap
                        )
                        over_run_cap = bool(
                            args.max_run_cost_usd
                            and projected_run_cost > args.max_run_cost_usd
                        )

                        final_code = initial_code
                        final_extraction_error = initial_extraction_error
                        final_execution = initial_execution
                        repaired = False
                        if not initial_ok and not over_instance_cap and not over_run_cap:
                            print(f"[{iid}] Stage 3/3: one code repair...", flush=True)
                            try:
                                repair_response, _ = logged_model_request(
                                    client, spec, rec, "code_repair",
                                    build_repair_prompt(
                                        query, formulation, initial_code,
                                        initial_execution,
                                    ),
                                    REPAIR_SYSTEM_INSTRUCTIONS,
                                )
                            except Exception as exc:
                                rec.set(repair_attempted=True)
                                rec.fail(exc, "single code repair request")
                                continue
                            _, repaired_code, repair_extraction_error = (
                                CORE.extract_code_from_response(repair_response)
                            )
                            if repaired_code and not repair_extraction_error:
                                repaired_code = CORE.replace_file_paths_in_code(
                                    repaired_code, [str(p) for p in DATA_FILES]
                                )
                            print(f"[{iid}] Local execution 2/2...", flush=True)
                            repair_execution, repair_ok = execute_local_attempt(
                                rec, "repair_local_execution", repaired_code,
                                repair_extraction_error, args.timeout,
                            )
                            rec.set(
                                repair_attempted=True,
                                repair_output=repair_response,
                                repair_code_output=repaired_code,
                                repair_extraction_error=repair_extraction_error,
                                repair_execution=repair_execution,
                            )
                            final_code = repaired_code
                            final_extraction_error = repair_extraction_error
                            final_execution = repair_execution
                            repaired = repair_ok

                        rec.set(
                            model_output=code_response,
                            code_output=final_code,
                            extraction_error=final_extraction_error,
                            final_execution=final_execution,
                            repaired_successfully=repaired,
                        )
                        if not final_execution.get("ok"):
                            rec.fail(
                                RuntimeError(final_execution.get("error")
                                             or "final local execution failed"),
                                "two-stage workflow",
                            )

                    current_instance_cost = rec.summary()["cost_usd"]
                    projected_run_cost = logger.spent_usd() + current_instance_cost
                    if instance_cap and current_instance_cost > instance_cap:
                        instance_cost_stop = (
                            f"instance {iid} cost ${current_instance_cost:.4f}, over "
                            f"its ${instance_cap:.2f} "
                            f"{'fixed' if args.max_instance_cost_usd is not None else 'dynamic'} "
                            "instance limit"
                        )
                        rec.fail(RuntimeError(instance_cost_stop), "cost safety stop")
                    if (args.max_run_cost_usd
                            and projected_run_cost > args.max_run_cost_usd):
                        safety_stop = (
                            f"run cost ${projected_run_cost:.4f} exceeded the "
                            f"${args.max_run_cost_usd:.2f} limit"
                        )
                        rec.fail(RuntimeError(safety_stop), "run cost safety stop")
            except lx.BudgetExceeded as exc:
                safety_stop = str(exc)
                break
            if instance_cost_stop:
                print(f"INSTANCE COST LIMIT: {instance_cost_stop}; continuing")
            if safety_stop:
                break
    finally:
        summary = logger.close()

    if safety_stop:
        summary["safety_stop"] = safety_stop
        summary_path = logger.dir_path / "summary.json"
        if summary_path.exists():
            summary_path.write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        print(f"SAFETY STOP: {safety_stop}")
    print(json.dumps(summary, indent=2))
    if not args.no_report:
        run_report(logger.dir_path, timeout=args.timeout)
    print(f"run directory: {logger.dir_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
