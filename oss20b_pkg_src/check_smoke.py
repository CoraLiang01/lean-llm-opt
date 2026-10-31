#!/usr/bin/env python3
"""
check_smoke.py
==============
Grade a smoke run against the criteria that actually matter for this port.

    python check_smoke.py large_scale_or
    python check_smoke.py air_nrm_ls
    python check_smoke.py loto

Reads the newest run under <task>/runs/ and answers the only question worth
asking after a smoke test: did the pipeline work, or did it merely not crash?

The two failures this exists to catch are both silent -- the run completes,
the tables fill in, and the numbers are wrong:

  * Ollama's context window defaults to a few thousand tokens on many builds.
    When the window is short Ollama does not error, it drops the front of the
    prompt -- which is where the retrieved examples and the pasted data tables
    live. The model then answers a question it was only shown the end of.

  * gpt-oss emits chain-of-thought. If the server does not hand it back in a
    separate field it arrives glued to the answer, and a classifier that
    compares the reply against a fixed list matches nothing and falls through
    to its catch-all for every instance.

Exit code is the number of failed checks.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Per-task expectations, measured on the GPT-4.1 arm of the same benchmarks.
# `prompt_floor` is per instance, summed over that instance's calls, and is
# roughly a third of what GPT-4.1 needed -- low enough not to fire on normal
# variation, high enough that a truncated context cannot hide under it.
EXPECT = {
    "large_scale_or": {
        "dir": "large_scale_or",
        # GPT-4.1: median 15.3k per instance across 101 instances.
        "prompt_floor": 8000, "prompt_soft": 12000,
        # Every answer landing in the catch-all is the reasoning-leak signature.
        "catch_all": "Others with CSV",
        "needs_type": True,
    },
    "air_nrm_ls": {
        "dir": "air_nrm_ls",
        # GPT-4.1: median 34.5k per instance, max 47.5k. These queries carry
        # the whole fare/demand/attraction tables inline.
        "prompt_floor": 15000, "prompt_soft": 25000,
        "catch_all": None,
        # This pipeline routes to CA / Flow; a None means classification fell
        # over rather than chose.
        "needs_type": True,
    },
    "loto": {
        "dir": "loto",
        "prompt_floor": 8000, "prompt_soft": 12000,
        "catch_all": "Others with CSV",
        "needs_type": True,
    },
}


class Grader:
    def __init__(self) -> None:
        self.counts = {PASS: 0, WARN: 0, FAIL: 0}

    def report(self, level: str, name: str, detail: str = "") -> None:
        self.counts[level] += 1
        print(f"  [{level}] {name}", flush=True)
        if detail:
            for line in str(detail).splitlines():
                print(f"         {line}", flush=True)


def newest_run(runs: Path) -> Path | None:
    found = [p for p in runs.glob("*") if (p / "instances.jsonl").exists()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def grade(task: str, out_dir: str = "runs") -> int:
    """Grade the newest run of `task`. Returns the number of FAILs."""
    spec = EXPECT.get(task)
    if spec is None:
        print(f"unknown task: {task}", file=sys.stderr)
        return 1

    runs = ROOT / spec["dir"] / out_dir
    run = newest_run(runs) if runs.exists() else None
    if run is None:
        print(f"no completed run under {runs} -- run the smoke test first",
              file=sys.stderr)
        return 1

    g = Grader()
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (run / "instances.jsonl").open(encoding="utf-8")]
    calls = [c for r in rows for c in r.get("calls", []) if c.get("type") == "llm"]

    coder = (cfg.get("models") or {}).get("coder") or \
            (cfg.get("models") or {}).get("modeler") or {}
    max_tokens = coder.get("max_tokens") or 32768
    holdout = (cfg.get("classification") or {}).get("holdout_type")

    print("=" * 68)
    print(f"  smoke check: {run.name}")
    print(f"  profile {cfg.get('model_profile')}   model {coder.get('model')}")
    print(f"  endpoint {coder.get('base_url')}   num_ctx "
          f"{(coder.get('extra') or {}).get('num_ctx', '<unset>')}")
    print(f"  dataset {cfg.get('dataset')} -> {cfg.get('dataset_path')}")
    if holdout:
        print(f"  LOTO holdout: {holdout}")
    print("=" * 68, flush=True)

    # ---- 1. did every instance finish ------------------------------------
    if not rows:
        g.report(FAIL, "no instances logged")
        return g.counts[FAIL]
    ok = [r for r in rows if r.get("status") == "ok"]
    if len(ok) == len(rows):
        g.report(PASS, f"{len(rows)}/{len(rows)} instances completed")
    else:
        bad = [(r.get("instance_id"), (r.get("errors") or ["?"])[0][:100])
               for r in rows if r.get("status") != "ok"]
        g.report(FAIL, f"{len(rows) - len(ok)}/{len(rows)} instances failed",
                 "\n".join(f"instance {i}: {e}" for i, e in bad))

    # ---- 2. prompt size: the num_ctx canary ------------------------------
    prompts = [r["summary"]["prompt_tokens"] for r in rows
               if r["summary"].get("prompt_tokens")]
    if prompts:
        med = statistics.median(prompts)
        span = f"median {med:,.0f}, range {min(prompts):,}-{max(prompts):,}"
        if med < spec["prompt_floor"]:
            g.report(FAIL, f"prompt tokens far below expectation ({span})",
                     f"the GPT-4.1 arm of this benchmark used well over "
                     f"{spec['prompt_floor']:,} tokens per instance. A median "
                     f"this low means Ollama truncated the front of the "
                     f"prompt: num_ctx is not in effect.\n"
                     f"check `num_ctx: 65536` under the gpt-oss-20b profile in "
                     f"exp_config.yaml, and that the server was restarted "
                     f"since any change")
        elif med < spec["prompt_soft"]:
            g.report(WARN, f"prompt tokens lower than expected ({span})",
                     "not conclusive on a couple of instances, but worth "
                     "watching if the full run stays here")
        else:
            g.report(PASS, f"prompt tokens in the expected range ({span})")

    # ---- 3. classification: the reasoning-leak canary --------------------
    preds = [r.get("pred_type") for r in rows]
    raw = [r.get("pred_type_raw") for r in rows if r.get("pred_type_raw")]
    distinct = {p for p in preds if p}
    dist = ", ".join(f"{p}x{preds.count(p)}" for p in sorted(distinct, key=str))
    catch_all = spec["catch_all"]
    if spec["needs_type"] and not distinct:
        g.report(FAIL, "no instance produced a problem type",
                 "classification returned nothing -- the usual cause is "
                 "reasoning text arriving inside `content` and matching none "
                 "of the expected labels. Check `reasoning: true`."
                 + (f"\nraw classifier replies: {raw[:3]}" if raw else ""))
    elif catch_all and distinct == {catch_all} and len(rows) > 1:
        g.report(FAIL, f"every instance classified as '{catch_all}'",
                 f"predictions: {dist}\n"
                 f"the classifier falls back to this label when the reply "
                 f"matches nothing, so reasoning text glued to the answer "
                 f"produces exactly this.\n"
                 f"check `reasoning: true` under the gpt-oss-20b profile; if "
                 f"this Ollama build rejects `think`, say so before the full "
                 f"run -- the parser needs changing, not the config"
                 + (f"\nraw classifier replies: {raw[:3]}" if raw else ""))
    else:
        g.report(PASS, f"classifier produced {len(distinct)} distinct type(s)",
                 f"predictions: {dist}")

    # ---- 4. output truncation -------------------------------------------
    comps = [r["summary"]["completion_tokens"] for r in rows
             if r["summary"].get("completion_tokens")]
    if comps and max(comps) >= 0.98 * max_tokens:
        g.report(FAIL, f"output hit the {max_tokens:,} token ceiling",
                 "generated code is cut off mid-statement; raise max_tokens")
    elif comps:
        g.report(PASS, f"no output truncation "
                       f"(max {max(comps):,} of {max_tokens:,})")

    # ---- 5. token accounting --------------------------------------------
    estimated = [c for c in calls if c.get("token_source") == "estimated"]
    if not calls:
        g.report(WARN, "no LLM calls recorded")
    elif estimated:
        g.report(WARN, f"{len(estimated)}/{len(calls)} calls had tokens estimated",
                 "the server returned no usage counts, so token and cost "
                 "columns are tiktoken estimates and must be labelled as such")
    else:
        g.report(PASS, f"all {len(calls)} LLM calls reported real token counts")

    # ---- 6. stage instrumentation ---------------------------------------
    stages = set()
    for r in rows:
        stages |= set(r["summary"].get("by_stage", {}))
    if stages:
        g.report(PASS, f"{len(stages)} stage(s) instrumented",
                 f"stages seen: {', '.join(sorted(stages))}")
    else:
        g.report(WARN, "no per-stage breakdown recorded")

    # ---- 7. the run really was local ------------------------------------
    cost = sum(r["summary"].get("cost_usd") or 0 for r in rows)
    if cfg.get("model_profile") == "gpt-oss-20b" and cost > 0:
        g.report(WARN, f"non-zero API cost logged (${cost:.4f})",
                 "a local run should cost nothing; check which endpoint answered")
    else:
        g.report(PASS, "local model, no API cost" if not cost
                 else f"API cost ${cost:.4f}")

    # ---- 8. code actually came out --------------------------------------
    empty = [r.get("instance_id") for r in rows
             if not (r.get("code_output") or "").strip()]
    if empty:
        g.report(FAIL, f"{len(empty)} instance(s) produced no gurobipy code",
                 f"instance ids: {empty}\n"
                 "without code there is nothing for the scorer to run")
    else:
        g.report(PASS, "every instance produced gurobipy code")

    # ---- 9. LOTO: the fold has to be sealed -----------------------------
    if holdout:
        named = [r.get("instance_id") for r in rows
                 if holdout.lower() in str(r.get("pred_type_raw")
                                           or r.get("pred_type") or "").lower()]
        routed = [r.get("instance_id") for r in rows
                  if holdout.lower() in str(r.get("routed_to") or "").lower()]
        if named or routed:
            g.report(FAIL, "the held-out type was not fully removed",
                     f"named by the classifier: {named}\n"
                     f"routed to its own workflow: {routed}\n"
                     "this fold's numbers cannot be used")
        else:
            g.report(PASS, f"fold sealed: '{holdout}' never named, never "
                           f"routed to")
        dest: dict = {}
        for r in rows:
            dest[r.get("routed_to")] = dest.get(r.get("routed_to"), 0) + 1
        g.report(PASS, "routing after removal",
                 ", ".join(f"{k}: {v}" for k, v in dest.items()))

    print("\n" + "=" * 68)
    print(f"  {g.counts[PASS]} PASS   {g.counts[WARN]} WARN   {g.counts[FAIL]} FAIL")
    if g.counts[FAIL]:
        print("\n  Do NOT start the full runs. Send this output back.")
    else:
        print("\n  Smoke looks good. Send this output plus the packed runs")
        print("  before starting the full runs.")
    print("=" * 68, flush=True)
    return g.counts[FAIL]


if __name__ == "__main__":
    sys.exit(grade(sys.argv[1] if len(sys.argv) > 1 else "large_scale_or",
                   sys.argv[2] if len(sys.argv) > 2 else "runs"))
