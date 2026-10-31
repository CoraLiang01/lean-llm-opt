#!/usr/bin/env python3
"""Estimate Gemini 3 Pro (3.0) resources from Gemini 3.1 Pro observations.

Offline, standard library only. With no arguments, processes the five runs in
gemini_3pro_cost_estimate.md. Original logs are never modified.

    python3 estimate_gemini_3pro.py
    python3 estimate_gemini_3pro.py --run-dir runs/NAME --thinking-ratios 1 1.25
    python3 estimate_gemini_3pro.py --total-tokens 810416 --api-cost 4.896942

Gemini visible completion tokens EXCLUDE thinking in this project's logs.
Cached tokens are a SUBSET of prompt tokens. Total = prompt + visible + thinking.
Scaling factors are explicit assumptions, not fitted cross-model coefficients.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-pro-preview"
TARGET_PRICE_SOURCE = "https://blog.google/innovation-and-ai/technology/developers-tools/gemini-3-developers/"
PAPER_RUNS = (
    "Base-SingleCall__gemini-3.1-pro__Large-Scale-OR__r0__20260918-231948__0f8513964a8c",
    "Base-SingleCall__gemini-3.1-pro__Air-NRM-CA-LS__r0__20260826-153258__031cd677f6e7",
    "Base-SingleCall__gemini-3.1-pro__Air-NRM-NP-LS__r0__20260913-000117__b440c95010af",
    "Base-Agentic-2Stage-1Repair__gemini-3.1-pro__Air-NRM-CA-LS__r0__20260912-181212__86e0c2eb78a8",
    "Base-Agentic-2Stage-1Repair__gemini-3.1-pro__Air-NRM-NP-LS__r0__20260912-192513__c59e939dfdea",
)


def nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


@dataclass(frozen=True)
class Usage:
    prompt_tokens: float
    completion_tokens: float  # visible output only
    reasoning_tokens: float = 0
    cached_prompt_tokens: float = 0

    def __post_init__(self):
        for key, value in asdict(self).items():
            nonnegative(value, key)
        if self.cached_prompt_tokens > self.prompt_tokens:
            raise ValueError("cached_prompt_tokens cannot exceed prompt_tokens")

    @property
    def total_tokens(self) -> float:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    @classmethod
    def from_call(cls, call: dict) -> Usage:
        required = ("prompt_tokens", "completion_tokens", "reasoning_tokens")
        for key in required:
            if key not in call or call[key] is None:
                raise ValueError(f"missing {key}; cannot treat unknown usage as zero")
        values = {key: call[key] for key in required}
        values["cached_prompt_tokens"] = call.get("cached_prompt_tokens", 0)
        for key, value in values.items():
            number = nonnegative(value, key)
            if not number.is_integer():
                raise ValueError(f"observed {key} must be an integer")
            values[key] = int(number)
        usage = cls(**values)
        if call.get("total_tokens") is not None:
            total = nonnegative(call["total_tokens"], "total_tokens")
            if total != usage.total_tokens:
                raise ValueError("total_tokens must equal prompt + visible completion + reasoning")
        return usage


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens; frozen project schedule, no storage/tool charges.

    The 3.0 standard $2/$12 rates are documented by its launch announcement.
    Cache and long-context prices below reuse the 3.1 schedule as an explicit
    proxy assumption. All five paper runs stay within the standard tier.
    """
    input_per_million: float = 2.0
    cached_input_per_million: float = 0.2
    output_per_million: float = 12.0
    long_input_per_million: float = 4.0
    long_cached_input_per_million: float = 0.4
    long_output_per_million: float = 18.0
    long_context_threshold: int = 200_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            nonnegative(value, name)

    def cost(self, usage: Usage) -> float:
        # The tier is determined by EACH request's entire prompt, cache included.
        if usage.prompt_tokens > self.long_context_threshold:
            pin, pcache, pout = (self.long_input_per_million,
                                self.long_cached_input_per_million,
                                self.long_output_per_million)
        else:
            pin, pcache, pout = (self.input_per_million,
                                self.cached_input_per_million,
                                self.output_per_million)
        return ((usage.prompt_tokens - usage.cached_prompt_tokens) * pin
                + usage.cached_prompt_tokens * pcache
                + (usage.completion_tokens + usage.reasoning_tokens) * pout) / 1_000_000


def estimate_call(usage: Usage, thinking_ratio: float = 1.0,
                  preserve_cache: bool = True,
                  pricing: Pricing = Pricing()) -> dict:
    """Fixed input/visible output/call trajectory, configurable target thinking."""
    ratio = nonnegative(thinking_ratio, "thinking_ratio")
    target = Usage(usage.prompt_tokens, usage.completion_tokens,
                   usage.reasoning_tokens * ratio,
                   usage.cached_prompt_tokens if preserve_cache else 0)
    return {**asdict(target), "total_tokens": target.total_tokens,
            "token_cost_usd": pricing.cost(target)}


def estimate_totals(total_tokens: int, api_cost: float,
                    token_ratio: float = 1.0, cost_ratio: float = 1.0) -> dict:
    """Explicit proportional proxy when only two observed totals are available.

    Cost and token ratios are independent. No thinking/cache/tier breakdown is
    inferred from aggregate totals, so no component repricing is possible here.
    """
    total = nonnegative(total_tokens, "total_tokens")
    if not total.is_integer():
        raise ValueError("observed total_tokens must be an integer")
    source_cost = nonnegative(api_cost, "api_cost")
    token_factor = nonnegative(token_ratio, "token_ratio")
    cost_factor = nonnegative(cost_ratio, "cost_ratio")
    return {
        "source_total_tokens": int(total),
        "source_recorded_api_cost_usd": source_cost,
        "token_ratio": token_factor,
        "cost_ratio": cost_factor,
        "estimated_total_tokens": total * token_factor,
        "estimated_api_cost_usd": source_cost * cost_factor,
        "estimate_method": "aggregate_proportional_proxy",
    }


def load_run(directory: Path) -> tuple[dict, list[Usage], dict]:
    """Read only LLM usage and metadata; never export prompts or responses."""
    directory = directory.resolve()
    calls_path = directory / "calls.jsonl"
    calls, usages = [], []
    with calls_path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            call = json.loads(line)
            if not isinstance(call, dict):
                raise ValueError(f"{calls_path}:{line_no}: expected a JSON object")
            if call.get("type") != "llm":
                continue  # tool records are not billable Gemini token requests
            if call.get("provider") != "google" or not str(call.get("model", "")).startswith("gemini-3.1-pro"):
                raise ValueError(f"{calls_path}:{line_no}: expected Gemini 3.1 Pro LLM usage")
            try:
                usage = Usage.from_call(call)
            except ValueError as exc:
                raise ValueError(f"{calls_path}:{line_no}: {exc}") from exc
            usages.append(usage)
            calls.append(call)
    if not calls:
        raise ValueError(f"no Gemini 3.1 Pro LLM records in {calls_path}")
    observed = {key: sum(getattr(u, key) for u in usages) for key in asdict(usages[0])}
    observed["total_tokens"] = sum(u.total_tokens for u in usages)
    cost_present = all(c.get("cost_usd") is not None for c in calls)
    recorded_cost = (sum(nonnegative(c["cost_usd"], "cost_usd") for c in calls)
                     if cost_present else None)
    observed["recorded_api_cost_usd"] = recorded_cost
    observed["recomputed_token_cost_usd"] = sum(Pricing().cost(u) for u in usages)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for key, value in observed.items():
        summary_key = "total_" + key
        if summary_key in summary and summary[summary_key] != value:
            raise ValueError(f"{directory}: calls disagree with summary {summary_key}")
    if "total_n_llm_calls" in summary and summary["total_n_llm_calls"] != len(calls):
        raise ValueError(f"{directory}: LLM call count disagrees with summary")
    if recorded_cost is not None and "total_cost_usd" in summary:
        if not math.isclose(recorded_cost, summary["total_cost_usd"], abs_tol=1e-5):
            raise ValueError(f"{directory}: call costs disagree with summary total_cost_usd")
    observed["summary_recorded_api_cost_usd"] = summary.get("total_cost_usd")
    meta = {
        "source_run": summary.get("run_id", directory.name),
        "dataset": summary.get("dataset", calls[0].get("dataset", "")),
        "method": summary.get("method", calls[0].get("method", "")),
        "source_model": ", ".join(sorted({c["model"] for c in calls})),
        "source_n_instances": summary.get("n_instances"),
        "source_n_llm_calls": len(calls),
        "source_failed_calls": sum(c.get("ok") is False for c in calls),
        "source_unavailable_usage_calls": sum(c.get("token_source") == "unavailable" for c in calls),
        "source_long_context_calls": sum(u.prompt_tokens > 200_000 for u in usages),
        "source_calls_file": str(calls_path),
        "source_calls_sha256": hashlib.sha256(calls_path.read_bytes()).hexdigest(),
        "summary_present": bool(summary),
    }
    return meta, usages, observed


def estimate_run(directory: Path, thinking_ratios: Iterable[float] = (0.75, 1.0, 1.25),
                 preserve_cache: bool = True, pricing: Pricing = Pricing()) -> list[dict]:
    meta, usages, observed = load_run(Path(directory))
    result = []
    for ratio in thinking_ratios:
        scaled = [estimate_call(u, ratio, preserve_cache, pricing) for u in usages]
        row = {**meta, **{"source_" + k: v for k, v in observed.items()},
               "target_model": "gemini-3-pro-preview (3.0)",
               "thinking_ratio": nonnegative(ratio, "thinking_ratio"),
               "cache_mode": "preserve" if preserve_cache else "none",
               "estimate_method": "per_call_usage_proxy"}
        for key in asdict(usages[0]):
            row["estimated_" + key] = sum(s[key] for s in scaled)
        row["estimated_total_tokens"] = sum(s["total_tokens"] for s in scaled)
        row["estimated_api_cost_usd"] = sum(s["token_cost_usd"] for s in scaled)
        row["estimated_api_cost_usd_no_cache"] = sum(
            estimate_call(u, ratio, False, pricing)["token_cost_usd"] for u in usages)
        result.append(row)
    return result


def write_outputs(output_dir: Path, rows: list[dict], pricing: Pricing) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_model": "Gemini 3 Pro (3.0)",
        "resource_status": "estimated, not a Gemini 3.0 measurement",
        "assumptions": [
            "No fitted version-to-version token ratio; default ratios are proxy scenarios.",
            "Per-call mode fixes prompt, visible output, and request trajectory.",
            "Cache preservation is an assumption; the no-cache cost is also provided.",
            "Thinking ratios change hidden output only; cached input is not extra total tokens.",
            "Costs include token inference only, excluding tools, grounding and cache storage.",
            "Unavailable source usage cannot be recovered from logged zeros.",
        ],
        "default_per_call_pricing": asdict(pricing),
        "pricing_applied": all(r["estimate_method"] == "per_call_usage_proxy" for r in rows),
        "pricing_sources_last_checked": "2026-10-31",
        "pricing_basis": "Project frozen standard schedule; 3.0 cache/long-context rates use 3.1 as proxy.",
        "pricing_sources": [TARGET_PRICE_SOURCE, PRICE_SOURCE],
        "estimates": rows,
    }
    (output_dir / "estimates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output_dir / "estimates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, action="append", help="run directory containing calls.jsonl; repeatable")
    parser.add_argument("--total-tokens", type=int, help="aggregate-only mode: observed Gemini 3.1 total tokens")
    parser.add_argument("--api-cost", type=float, help="aggregate-only mode: observed Gemini 3.1 API cost in USD")
    parser.add_argument("--token-ratio", type=float, help="aggregate-only target/source token multiplier; default 1")
    parser.add_argument("--cost-ratio", type=float, help="aggregate-only target/source cost multiplier; default 1")
    parser.add_argument("--thinking-ratios", type=float, nargs="+", help="per-call hidden output multipliers; default 0.75 1 1.25")
    parser.add_argument("--no-cache", action="store_true", help="per-call target estimate assumes no cache hits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/gemini_3pro_estimate")
    args = parser.parse_args(argv)
    aggregate = args.total_tokens is not None or args.api_cost is not None
    if aggregate:
        if args.total_tokens is None or args.api_cost is None:
            parser.error("--total-tokens and --api-cost must be supplied together")
        if args.run_dir or args.thinking_ratios is not None or args.no_cache:
            parser.error("aggregate-only mode cannot use --run-dir, --thinking-ratios or --no-cache")
    elif args.token_ratio is not None or args.cost_ratio is not None:
        parser.error("--token-ratio/--cost-ratio require aggregate-only mode")
    try:
        if aggregate:
            rows = [estimate_totals(args.total_tokens, args.api_cost,
                                   args.token_ratio if args.token_ratio is not None else 1.0,
                                   args.cost_ratio if args.cost_ratio is not None else 1.0)]
            rows[0]["target_model"] = "gemini-3-pro-preview (3.0)"
        else:
            directories = args.run_dir or [ROOT / "runs" / name for name in PAPER_RUNS]
            missing = [str(d) for d in directories if not (d / "calls.jsonl").is_file()]
            if missing:
                raise ValueError(
                    "Run logs remain local and are not included in this code upload. "
                    "Use --run-dir with your own logs, or --total-tokens N --api-cost USD."
                )
            directories = list(dict.fromkeys(p.resolve() for p in directories))
            rows = [row for d in directories for row in estimate_run(
                d, args.thinking_ratios or (0.75, 1.0, 1.25), not args.no_cache)]
        write_outputs(args.output_dir, rows, Pricing())
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print("Gemini 3.0 Pro conditional estimates (no API calls)")
    for row in rows:
        label = f"{row.get('dataset', 'manual')} / {row.get('method', 'aggregate')}"
        scenario = f"thinking={row['thinking_ratio']:g}" if "thinking_ratio" in row else f"token_ratio={row['token_ratio']:g}, cost_ratio={row['cost_ratio']:g}"
        print(f"{label} | {scenario} | {row['estimated_total_tokens']:,.2f} tokens | ${row['estimated_api_cost_usd']:.6f}")
    print(f"Saved: {args.output_dir.resolve() / 'estimates.json'}")
    print(f"Saved: {args.output_dir.resolve() / 'estimates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
