# ReAct v7: final baseline-study results

Finalized UTC: 2026-10-06T07:05:32.094493+00:00

The five delivered ReAct notebooks were restored byte for byte from the archived v7 delivery. The full baseline reuses the complete existing v7 pass; each of the three requested studies contains 101 newly computed pipeline attempts. Only ablation classification predictions are reused. All errors, mismatches, truncations and timeouts remain in the 101-case denominator. No code/model repair or outcome-based rerun was used.

| Method | Classification correct | Optimal solve | Objective Match | Accuracy | Delta vs full |
|---|---:|---:|---:|---:|---:|
| ReAct v7 full | 94/101 | 96/101 | 94/101 | 93.07% | +0.00 pp |
| RAG Only | 94/101 | 90/101 | 86/101 | 85.15% | -7.92 pp |
| Few-shot Only | 94/101 | 75/101 | 63/101 | 62.38% | -30.69 pp |
| LOTO Examples And Route | 0/101 | 91/101 | 88/101 | 87.13% | -5.94 pp |

LOTO classification is scored against the original semantic label. Its original workflow is disabled, so 0/101 is expected under this definition. All 101 predictions were permitted by their folds; disabled-route executions were 0.

## Interpretation limits

Few-shot Only includes 12 confirmed failures of the existing v7 symbolic-model transfer filter: plain Objective:/Decision Variables:/Subject to: sections are not recognized by its Markdown/bold-only heading rule. Saved formulations contain an objective, but the transfer is rejected before any code-generation call. These failures are retained. The mathematical correctness of those formulations is not inferred, and their later code/solve outcomes are unknown. This is an implementation limitation that confounds a pure component-removal interpretation.

Additional Few-shot Only failures include one complete-source context rejection, one truncated model output, and the OR-084 1,800-second deadline. RAG Only includes nine truncated outputs; LOTO Examples And Route includes one. These are not excluded or regenerated. A single pass does not establish stability across repeated runs.

## Retry and data-tool audit

| Study | Repair | Pipeline rerun | Protocol restart | HTTP retry | CSVQA attempts | Complete-source fallback |
|---|---:|---:|---:|---:|---:|---:|
| RAG Only | 0 | 0 | 0 | 0 | 80 | 21 |
| Few-shot Only | 0 | 0 | 0 | 0 | 0 | 0 |
| LOTO Examples And Route | 0 | 0 | 0 | 0 | 52 | 7 |

The exact v7 architecture is retained: canonical NRM/RA/TP/AP/FLP workflows use ReAct with CSVQA; Others with CSV uses its v7 two-chain schema workflow. Consequently, CSVQA is not called in every v7 CSV case. Few-shot Only intentionally removes CSVQA and uses complete Python source Observations. Its generic audit reports two absent Observation traces on model-input/output failures; these are not uncertain CSVQA calls because the data-tool list is empty by design. Its complete-source reading is not counted as a fallback.

## Existing full pass, input/source identity and preserved history

Existing full v7 Objective Match: 101=94/101, Variants=34/36, redundant columns=289/315. Equal three-group mean=93.0866%; pooled matches=417/452. All nine redundancy sheets were checked. 100pct-S2=30/35 and 200pct-S3=31/35 remain disclosed soft-target shortfalls.

All 1,133 baseline input hashes were unchanged before and after the studies. Each study uses the same frozen full-base SHA, fixed model snapshot, source-only solver interface, Gurobi Threads=2, MIPGap=1e-4, 1,800-second case deadline, and rel_tol=abs_tol=1e-4. Runtime/library records are in ../v7_requested_studies_runtime_config.json.

The original direct-call files, later v8-v14 snapshots, old full/report records and pre-restoration v14 delivery were preserved. LOTO Examples Only was restored with the family but was not run by this corrected request. 606 remains disabled and unexecuted.

## Source notebooks and switches

- `LEAN_LLM_OPT_4.1_Large-scale_1006_ReAct.ipynb`: archived v7 source, SHA-256 `7bb36bb65963ecff267b1e1ee607b9128fe26390465290770f8cb255438ad266`.
- `Ablation_Study_Large_Scale_Or_RAG_Only_ReAct.ipynb`: archived v7 source, SHA-256 `b5ee97ff34d0550d7d9ae588bc2c1a0e49c9e1530ec7f73001089f374daa226b`.
- `Ablation_Study_Large_Scale_Or_Few-shot_Only_ReAct.ipynb`: archived v7 source, SHA-256 `81cb4ef9b2224a9f9d2240bd0310b1a117237453f774d9c7fd0d556f3a283d61`.
- `LOTO_Examples_Only_GPT4.1_Large-scale_ReAct.ipynb`: archived v7 source, SHA-256 `bf24f094d342aff10200aa0092c5a63a6c0f133dba01df31b177261cbad5342f`.
- `LOTO_Examples_And_Route_GPT4.1_Large-scale_ReAct.ipynb`: archived v7 source, SHA-256 `f9bde44956f943179e8ed5d9cb2443e6d39339df73fa29fa13875b2868da3902`.

All notebook inference switches remain off; use the controlled launcher for the measured thread/deadline configuration. `--baseline-study` permits only the three source-verified v7 studies explicitly requested by the user. The default improvement gate for other candidate versions remains unchanged. The report-only `--output-dir` option preserves historical report directories.

## Recorded reproduction commands

Run from the repository root with `/opt/miniconda3/envs/lean_llm_opt_4_1/bin/python`. The launcher preserves recorded successes and failures; it does not reattempt a scored case. Manifests, frozen notebooks/launchers, per-case logs, formulations, programs and results remain in the three separate method_v7 directories.

```bash
python scripts/evaluate_react_revision_20261006.py --method rag_only --version v7 --workers 3 --baseline-study --run
python scripts/evaluate_react_revision_20261006.py --method few_shot_only --version v7 --workers 3 --baseline-study --run
python scripts/evaluate_react_revision_20261006.py --method examples_and_route --version v7 --workers 3 --baseline-study --run
python scripts/report_react_revision_20261006.py --version v7 --output-dir outputs/react_revision_20261006/report_v7_reproduced_summary
```

Detailed row records: all_case_results.csv. Failed cases and stages: failures.csv. Paired changes against full: paired_changes.csv. Class/fold results: experiments_by_class.csv.
