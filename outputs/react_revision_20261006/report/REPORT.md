# ReAct architecture evaluation: v3

Status: full-model gate not yet passed.
606 enabled: False; executed: False.

NRM, RA, TP, AP and FLP restore the original ZERO_SHOT_REACT_DESCRIPTION agent.
The agent chooses CSVQA and receives its Observation before its Final Answer.
There is one agent invocation and one executed CSVQA call. Skipped tools, duplicate
requests and parsing failures are recorded; no protocol correction or model rerun exists.
The duplicate-request guard blocks a second extraction before it can repeat a planner call.
Few-shot Only uses the same ReAct agent class with an already supplied complete Python
Observation and no data tools. Others with CSV and the original query-only route remain.

All failed/error/truncated/timeout cases stay in the prescribed denominators.
Objective Match requires a finite objective from an optimal returned Gurobi model,
with unchanged rel_tol=abs_tol=1e-4. Classification and successful solve are separate.
A partial run is marked incomplete and is not a complete benchmark result.

## Full 101 by class

| class | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| AP | 5 | 0 | 0 | 0 | 0 |
| FLP | 14 | 3 | 3 | 3 | 3 |
| NRM | 25 | 25 | 25 | 24 | 24 |
| RA | 22 | 22 | 22 | 22 | 22 |
| TP | 9 | 9 | 9 | 9 | 9 |
| Others | 8 | 0 | 0 | 0 | 0 |
| Mixture | 18 | 1 | 0 | 1 | 1 |

## Full datasets and every sheet

| dataset | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| automatic | 101 | 60 | 59 | 59 | 59 |
| variants | 36 | 0 | 0 | 0 | 0 |
| columns/50pct-S1 | 35 | 0 | 0 | 0 | 0 |
| columns/50pct-S2 | 35 | 0 | 0 | 0 | 0 |
| columns/50pct-S3 | 35 | 0 | 0 | 0 | 0 |
| columns/100pct-S1 | 35 | 0 | 0 | 0 | 0 |
| columns/100pct-S2 | 35 | 0 | 0 | 0 | 0 |
| columns/100pct-S3 | 35 | 0 | 0 | 0 | 0 |
| columns/200pct-S1 | 35 | 0 | 0 | 0 | 0 |
| columns/200pct-S2 | 35 | 0 | 0 | 0 | 0 |
| columns/200pct-S3 | 35 | 0 | 0 | 0 | 0 |

## Ablations and LOTO against this ReAct baseline

| method | version | recorded | classification_correct | solved | objective_match | delta_matches_vs_full | delta_percentage_points | status |
|---|---|---|---|---|---|---|---|---|
| full | v3 | 60 | 59 | 59 | 59 | None | None | incomplete |
| rag_only |  | 0 |  |  |  |  |  | not_run |
| few_shot_only |  | 0 |  |  |  |  |  | not_run |
| examples_only |  | 0 |  |  |  |  |  | not_run |
| examples_and_route |  | 0 |  |  |  |  |  | not_run |


Ablations reuse only the newly validated ReAct full classification cache. LOTO
reclassifies after target-reference removal. Examples Only retains every route;
Examples And Route bans the target workflow and checks availability before modeling.
The original semantic-label accuracy is zero by design in the route-removal experiment.
Only fold grouping/exclusion and the scorer use the true class; no gold model/answer
is used to choose a replacement route or generate code.

## Failures

| method | failure_stage | count |
|---|---|---|
| full | execution | 1 |

| method | dataset | problem_id | failure_stage | failure_type | cause_certainty |
|---|---|---|---|---|---|
| full | automatic | OR-014 | execution | TypeError | confirmed; upstream semantic cause may be unresolved |


Unresolved objective mismatches remain unresolved until the saved model/program/source
records support an explanation. Missing CSVQA evidence is explicitly marked, and
fallback totals are lower bounds when evidence is unavailable. No trace is fabricated.

## Repairs, retries and fallbacks

| method | dataset | repair_count | pipeline_retry_count | http_retry_count | recorded_full_data_fallback_count | csvqa_evidence_unavailable_cases |
|---|---|---|---|---|---|---|
| full | automatic | 0 | 0 | 0 | 11 | 0 |
| full | variants | 0 | 0 | 0 | 0 | 0 |
| full | columns/50pct-S1 | 0 | 0 | 0 | 0 | 0 |
| full | columns/50pct-S2 | 0 | 0 | 0 | 0 | 0 |
| full | columns/50pct-S3 | 0 | 0 | 0 | 0 | 0 |
| full | columns/100pct-S1 | 0 | 0 | 0 | 0 | 0 |
| full | columns/100pct-S2 | 0 | 0 | 0 | 0 | 0 |
| full | columns/100pct-S3 | 0 | 0 | 0 | 0 | 0 |
| full | columns/200pct-S1 | 0 | 0 | 0 | 0 | 0 |
| full | columns/200pct-S2 | 0 | 0 | 0 | 0 | 0 |
| full | columns/200pct-S3 | 0 | 0 | 0 | 0 | 0 |


HTTP retries retain the original SDK transport policy. They are separate from model,
code and pipeline regeneration, all disabled. Each worker uses Gurobi Threads=2;
MIPGap=1e-4 and a common external 1,800-second deadline remain unchanged.

## Preserved controls and reproduction

The earlier direct-call results remain under outputs/minimal_revision_20261005.
Their 97/101 is not a verified score of the restored ReAct architecture. Current
notebooks are backed up before restoration under backups_direct_call_version.
The source-data extraction, exact identifiers, axis validation, generic solver
interfaces and immutable failed-case cache are retained. No question-ID, objective,
or reference-answer branch is added to generation.

Use /opt/miniconda3/envs/lean_llm_opt_4_1/bin/python with:

```bash
python scripts/evaluate_react_revision_20261006.py --method full --version v3 --workers 6 --run
python scripts/evaluate_react_revision_20261006.py --method rag_only --version v3 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method few_shot_only --version v3 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_only --version v3 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_and_route --version v3 --workers 3 --run
python scripts/report_react_revision_20261006.py
```

Every case/version is attempted once. Resuming preserves failures as well as successes.
Changed code or data requires a fresh result directory. Every manifest freezes the
notebook, input hashes, fixed model snapshot and execution settings. Source files,
model/code artifacts and logs are retained. 606 stays disabled until this revision
passes the required full-model gate and all nine sheets have been reviewed.
