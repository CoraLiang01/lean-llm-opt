#!/usr/bin/env python3
"""Run the 101-instance Large-Scale-OR single-call cost benchmark.

This is the Large-Scale-OR counterpart of Air-NRM's single-call benchmark.
Each instance makes exactly one tool-free model request containing the complete
contents of that row's Dataset_address files.  It records provider token usage,
one LLM call, estimated API cost, latency, the raw response, and extracted code
through the repository's unified RunLogger.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUERY_CSV = HERE / "Large-scale-or-101.csv"
PROFILES = {
    "gpt-5": "gpt-5",
    "gpt-4.1": "gpt-4.1",
    "openai": "gpt-5.2",
    "gpt-5.2": "gpt-5.2",
    "gemini": "gemini-3.5-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.1-pro": "gemini-3.1-pro",
    # Backward-compatible CLI alias. Gemini 3 Pro Preview was shut down;
    # Google directs API users to Gemini 3.1 Pro Preview instead.
    "gemini-3-pro": "gemini-3.1-pro",
}
EXPECTED_MODEL_IDS = {
    "gpt-5": "gpt-5",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-5.2": "gpt-5.2",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
}
GEMINI_DEFAULT_INSTANCE_COST_USD = 0.40

sys.path.insert(0, str(ROOT))
import leanopt_exp as lx  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE = load_module("large_scale_or_singlecall_core", HERE / "run_test_universe_agentic.py")
REPORT = load_module(
    "large_scale_or_singlecall_report",
    ROOT / "Test_Dataset" / "Air_NRM" / "large_scale" / "report_agentic_benchmark.py",
)


def parse_rows(value: str | None, n_rows: int) -> list[int]:
    """Parse zero-based row selectors such as ``0,2-4``."""
    if not value:
        return list(range(n_rows))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            if end < start:
                raise ValueError(f"invalid descending row range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    bad = sorted(i for i in selected if i < 0 or i >= n_rows)
    if bad:
        raise ValueError(f"row index outside 0..{n_rows - 1}: {bad}")
    return sorted(selected)


def parse_gold_objective(value) -> float:
    """Read numeric, currency-formatted, or Gurobi-log gold objectives."""
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    cleaned = re.sub(r"[,\s$¥€£]", "", text)
    try:
        return float(cleaned)
    except ValueError:
        pass
    match = re.search(
        r"optimal\s+objective\s*[:=]?\s*"
        r"([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1).replace(",", ""))
    raise ValueError(f"invalid Label-objective: {value!r}")


def resolve_data_files(value) -> list[Path]:
    """Resolve one row's newline-delimited Dataset_address within the repo."""
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Dataset_address is empty")
    repo_root = ROOT.resolve()
    files: list[Path] = []
    for raw in str(value).splitlines():
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"Dataset_address escapes repository root: {raw}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"dataset file not found: {path}")
        files.append(path)
    if not files:
        raise ValueError("Dataset_address contains no files")
    return files


def call_cost(metadata: dict, model_spec: lx.ModelSpec) -> dict:
    """Calculate request cost using the frozen prices in exp_config.yaml."""
    prompt = int(metadata.get("prompt_tokens") or 0)
    cached = min(prompt, int(metadata.get("cached_prompt_tokens") or 0))
    output = int(metadata.get("completion_tokens") or 0)
    # Gemini exposes hidden thinking separately; OpenAI output already includes
    # its reasoning-token subset.
    if model_spec.provider == "google":
        output += int(metadata.get("reasoning_tokens") or 0)
    extra = model_spec.extra or {}
    if prompt > int(extra.get("long_context_threshold", 10**18)):
        price_in = float(extra.get(
            "long_context_price_in_per_1m", model_spec.price_in_per_1m
        ))
        price_cached = float(extra.get(
            "long_context_price_cached_in_per_1m",
            model_spec.price_cached_in_per_1m,
        ))
        price_out = float(extra.get(
            "long_context_price_out_per_1m", model_spec.price_out_per_1m
        ))
        price_tier = "long_context"
    else:
        price_in = model_spec.price_in_per_1m
        price_cached = model_spec.price_cached_in_per_1m
        price_out = model_spec.price_out_per_1m
        price_tier = "standard"
    token_cost = (
        (prompt - cached) * price_in
        + cached * price_cached
        + output * price_out
    ) / 1_000_000
    return {
        "cost_usd": round(token_cost, 8),
        "token_cost_usd": round(token_cost, 8),
        "tool_cost_usd": 0.0,
        "price_tier": price_tier,
        "price_in_per_1m": price_in,
        "price_cached_in_per_1m": price_cached,
        "price_out_per_1m": price_out,
    }


def build_client_config(cfg: lx.ExpConfig, args) -> dict:
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
        "reasoning_effort": (spec.extra or {}).get("reasoning_effort"),
        "if_proxy": args.if_proxy,
        "proxy_url": args.proxy_url,
    }


def prompt_for_api_key(model_spec: lx.ModelSpec) -> None:
    provider = "Gemini" if model_spec.provider == "google" else "OpenAI"
    key = getpass.getpass(f"Enter {provider} API key (input hidden): ").strip()
    if len(key) < 20:
        raise ValueError(f"{provider} API key looks empty or incomplete")
    os.environ[model_spec.api_key_env] = key
    print(f"{provider} API key accepted: ****{key[-4:]}")


def instance_cost_limit(row, all_rows: pd.DataFrame, args) -> float:
    """Allocate 25% equally and 75% in proportion to the input Token field."""
    if args.max_instance_cost_usd is not None:
        return args.max_instance_cost_usd
    if not args.max_run_cost_usd:
        return 0.0
    token_weight = max(float(row["Token"]), 0.0)
    total_weight = max(float(all_rows["Token"].clip(lower=0).sum()), 1.0)
    return (
        0.25 * args.max_run_cost_usd / len(all_rows)
        + 0.75 * args.max_run_cost_usd * token_weight / total_weight
    )


def logged_model_request(client, spec: lx.ModelSpec, rec, prompt: str,
                         data_files: list[Path]) -> tuple[str, dict]:
    """Make and fully log exactly one provider request."""
    started = time.perf_counter()
    try:
        metadata = client.generate_with_metadata(
            prompt,
            [str(path) for path in data_files],
            system_instructions=CORE.SYSTEM_INSTRUCTIONS,
        )
        raw_text = metadata.pop("text", "")
        prices = call_cost(metadata, spec)
        rec.add_call({
            "type": "llm",
            "stage": "single_call",
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
            "type": "llm",
            "stage": "single_call",
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
            "token_source": "unavailable",
            "latency_s": round(time.perf_counter() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def is_fatal_provider_error(exc: Exception) -> bool:
    """Return True when retrying every remaining row cannot fix the error."""
    message = str(exc).lower()
    fatal_markers = (
        "404 not_found",
        "not found for api version",
        "401 unauthorized",
        "403 forbidden",
        "unauthenticated",
        "permission_denied",
        "api key not valid",
        "invalid api key",
        "resource_exhausted",
        "quota exceeded",
    )
    return any(marker in message for marker in fatal_markers)


def validate_dataset(data: pd.DataFrame) -> list[list[Path]]:
    required = {
        "Type by size", "Token", "Problem Type", "Query",
        "Dataset_address", "Label-objective",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{QUERY_CSV} is missing columns: {sorted(missing)}")
    if len(data) != 101:
        raise ValueError(f"expected 101 benchmark rows, found {len(data)}")
    resolved = [resolve_data_files(row["Dataset_address"])
                for _, row in data.iterrows()]
    for value in data["Label-objective"]:
        parse_gold_objective(value)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile", choices=sorted(PROFILES),
        help=("gpt-5, gpt-4.1, gpt-5.2/openai, gemini-3.5-flash/gemini, "
              "or gemini-3.1-pro"),
    )
    parser.add_argument("--rows", help="zero-based rows, e.g. 0,2-4")
    parser.add_argument("--limit", type=int,
                        help="run only the first N selected rows")
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-instance solver timeout used by the final report")
    parser.add_argument("--max-instance-cost-usd", type=float, default=None,
                        help=("fixed per-instance cap; Gemini defaults to $0.40, "
                              "other profiles allocate the run cap by Token"))
    parser.add_argument("--max-run-cost-usd", type=float, default=50.0,
                        help="stop after total API cost reaches this value (default: 50)")
    proxy = parser.add_mutually_exclusive_group()
    proxy.add_argument("--if-proxy", "--if_proxy", dest="if_proxy",
                       action="store_true", help="enable proxy explicitly")
    proxy.add_argument("--no-proxy", "--no_proxy", dest="if_proxy",
                       action="store_false", help="use a direct connection (default)")
    parser.set_defaults(if_proxy=False)
    parser.add_argument("--proxy-url", default="http://127.0.0.1:7897")
    parser.add_argument("--prompt-key", action="store_true",
                        help="securely prompt for this run's API key")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate all 101 rows without API calls or output files")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be nonnegative")
    if ((args.max_instance_cost_usd is not None
         and args.max_instance_cost_usd < 0)
            or args.max_run_cost_usd < 0):
        parser.error("cost limits must be nonnegative (0 disables that limit)")

    profile = PROFILES[args.profile]
    if (profile in {"gemini-3.1-pro", "gemini-3.5-flash"}
            and args.max_instance_cost_usd is None):
        args.max_instance_cost_usd = GEMINI_DEFAULT_INSTANCE_COST_USD
    if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    cfg = lx.load_config(
        ROOT / "exp_config.yaml",
        model_profile=profile,
        method="Base-SingleCall",
        dataset="Large-Scale-OR",
        repeat_index=args.repeat_index,
        solver_time_limit_s=args.timeout,
        budget_usd_per_run=args.max_run_cost_usd,
    )
    spec = cfg.spec("modeler")
    expected_model = EXPECTED_MODEL_IDS[profile]
    if spec.model != expected_model:
        raise ValueError(
            f"profile {profile!r} resolves to {spec.model!r}; "
            f"expected {expected_model!r}"
        )
    if args.profile == "gemini-3-pro":
        print(
            "NOTE: Gemini 3 Pro Preview has been shut down; using its official "
            "replacement, Gemini 3.1 Pro Preview."
        )
    cfg.manifest["request_policy"] = {
        "workflow": "single_call",
        "model_requests_per_query": 1,
        "local_execution_attempts_during_generation": 0,
        "sdk_retries": 0,
        "hosted_execution_tools": False,
        "data_delivery": "complete_inline_csv_text_from_Dataset_address",
        "gold_objective_visible_to_model": False,
        "preserve_output_when_code_invalid": True,
    }
    cfg.manifest["cost_safety"] = {
        "max_instance_cost_usd": args.max_instance_cost_usd,
        "max_run_cost_usd": args.max_run_cost_usd,
        "instance_cost_policy": (
            "fixed_override" if args.max_instance_cost_usd is not None
            else "25pct_equal_overhead_plus_75pct_Token_weight"
        ),
        "enforcement": "post_response",
    }

    if not QUERY_CSV.is_file():
        raise FileNotFoundError(QUERY_CSV)
    data = pd.read_csv(QUERY_CSV)
    resolved_files = validate_dataset(data)
    rows = parse_rows(args.rows, len(data))
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("no benchmark rows selected")
    # Cost shares must be calculated over this invocation's selected rows.
    # Using all 101 rows made a one-row smoke test receive only ~1/101 of its
    # stated run budget and incorrectly marked a successful response failed.
    selected_data = data.iloc[rows]

    print(f"profile={profile}  provider={spec.provider}  model={spec.model}")
    print(f"method=Base-SingleCall  dataset=Large-Scale-OR")
    print(f"query_file={QUERY_CSV.relative_to(ROOT)}  validated_rows={len(data)}")
    print(f"selected_instances={len(rows)}  rows={rows}")
    print(
        "data_files_per_instance="
        f"{min(map(len, resolved_files))}..{max(map(len, resolved_files))}"
    )
    if args.max_instance_cost_usd is None:
        limits = [instance_cost_limit(row, selected_data, args)
                  for _, row in selected_data.iterrows()]
        print(
            "cost_limits=Token-weighted "
            f"${min(limits):.4f}..${max(limits):.4f}/instance, "
            f"${args.max_run_cost_usd:.2f}/run"
        )
    else:
        print(
            f"cost_limits=${args.max_instance_cost_usd:.4f}/instance, "
            f"${args.max_run_cost_usd:.2f}/run"
        )
    if args.dry_run:
        print("dry-run passed; no API call or run directory was created")
        return 0

    CORE.setup_proxy({"if_proxy": args.if_proxy, "proxy_url": args.proxy_url})
    if args.prompt_key:
        prompt_for_api_key(spec)
    else:
        lx.ensure_api_keys(cfg, interactive=False)
    client = CORE.create_client(build_client_config(cfg, args))
    logger = lx.RunLogger(cfg)
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
            files = resolved_files[iid]
            query = str(row["Query"])
            instance_cap = instance_cost_limit(row, selected_data, args)
            relative_addresses = "\n".join(
                str(path.relative_to(ROOT.resolve())) for path in files
            )
            instance_cost_stop = ""
            try:
                with logger.instance(
                    iid,
                    query=query,
                    gold_type=str(row["Problem Type"]),
                    gold_objective=parse_gold_objective(row["Label-objective"]),
                    dataset_address=relative_addresses,
                    size_class=str(row["Type by size"]),
                    estimated_input_tokens=int(row["Token"]),
                    cost_limit_usd=round(instance_cap, 6),
                ) as rec:
                    prompt = CORE.build_prompt(query)
                    request_ok = False
                    try:
                        raw_text, _ = logged_model_request(
                            client, spec, rec, prompt, files
                        )
                    except Exception as exc:
                        rec.fail(exc, "provider request")
                        if is_fatal_provider_error(exc):
                            safety_stop = (
                                f"fatal provider error on instance {iid}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    else:
                        request_ok = True

                    if request_ok:
                        math_model, code, extraction_error = (
                            CORE.extract_code_from_response(raw_text)
                        )
                        if code and not extraction_error:
                            code = CORE.replace_file_paths_in_code(
                                code, [str(path) for path in files]
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

                        current_cost = rec.summary()["cost_usd"]
                        projected_run_cost = logger.spent_usd() + current_cost
                        if instance_cap and current_cost > instance_cap:
                            instance_cost_stop = (
                                f"instance {iid} cost ${current_cost:.4f}, over its "
                                f"${instance_cap:.4f} limit"
                            )
                            rec.fail(
                                RuntimeError(instance_cost_stop),
                                "cost safety stop",
                            )
                        if (args.max_run_cost_usd
                                and projected_run_cost > args.max_run_cost_usd):
                            safety_stop = (
                                f"run cost ${projected_run_cost:.4f} exceeded the "
                                f"${args.max_run_cost_usd:.2f} limit"
                            )
                            rec.fail(
                                RuntimeError(safety_stop),
                                "run cost safety stop",
                            )
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
    if not args.no_report and summary.get("n_instances", 0):
        REPORT.generate_report(logger.dir_path, timeout=args.timeout, diagnose=False)
    print(f"run directory: {logger.dir_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
