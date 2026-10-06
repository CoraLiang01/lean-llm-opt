# GPT-4.1 V1 LOTO result audit — 2026-10-04

## Evidence and limits

The current original result CSVs are access-protected. They were not decrypted or bypassed. This audit freshly reads all 202 per-case `solution.json` files and 14 `fold_manifest.json` files. Reference objective values and predicted classifications come from the previously verified `outputs/leave_one_type_out/gpt41/V1_review_summary/V1_analysis.json`. All current solution statuses, objectives, and errors agree with those previous records. Classification matrices can therefore be reproduced from the prior records, but were not independently re-read from the protected original CSVs today. No generated program or LLM was executed; original notebooks/results were not modified.

Each experiment has seven folds, 101 unique IDs, and exactly one solution file per expected problem. Manifests identify `gpt-4.1-2025-04-14` and identical reference and model-specific baseline SHA-256 hashes. Their reference removal and route availability metadata agree. All stored predicted labels/routes are allowed by their fold manifests.

| Experiment | Cases | Solved | Objective matches | Failed | Solved but mismatched | Classification correct |
|---|---:|---:|---:|---:|---:|---:|
| Examples only | 101 | 70 | 68 | 31 | 2 | 92 |
| Examples and route | 101 | 90 | 81 | 11 | 9 | Not applicable |

## Paired decomposition

| Transition from examples only to examples and route | Cases |
|---|---:|
| Matched in both | 58 |
| Newly matched | 23 |
| Lost match | 10 |
| Matched in neither | 10 |

The 23 gains comprise 15 former `NoneType` model-handle errors, four truncated model outputs, one missing model handle (`KeyError: 'model'`), one unsupported Gurobi keyword, one infeasible generated model, and one previously solved objective mismatch. Thus 22 gains recover execution failures and one corrects a previously computed objective. The ten losses comprise six computed objective mismatches and four execution failures (one each: ragged CSV, customer-column mismatch, numeric parsing, API timeout).

Execution transitions: 66 solve in both experiments, 24 recover from failure, four lose successful execution, and seven fail in both. On the same 66 successfully executed problems, examples-only matches 64/66 (96.97%) and examples-and-route matches 59/66 (89.39%). This descriptive subset is selected after treatment and is not an unbiased causal measure of mathematical-modeling capability.

## Execution failure categories

| Failure mechanism | Examples only | Examples and route |
|---|---:|---:|
| Global model handle is `None` | 19 | 0 |
| Missing global model handle | 1 | 0 |
| Model output truncated | 7 | 0 |
| Ragged legacy CSV observation | 2 | 7 |
| Unsupported Gurobi `keyformat` keyword | 1 | 0 |
| Infeasible generated model | 1 | 0 |
| Context length exceeded | 0 | 1 |
| Customer/column mapping failure | 0 | 1 |
| Numeric parsing failure | 0 | 1 |
| API timeout | 0 | 1 |

All 19 `NoneType` failures have statically verified generated functions that call `m.optimize()` but never `return m`; the global assignment `m = solve_...(...)` therefore receives `None`. The runner subsequently selects `namespace['m']` and enters `with model:`, causing the recorded exception. The optimizer was called, but the stored output has no objective/status proving optimality, so these cases must not be retrospectively scored correct without a validated rerun.

Representative evidence:

- `examples_only_V1/TP/results_cases/AUTO/OR-004/2d6eed870f5c7b23/solve.py`: function at line 4, optimize at line 27, assignment of implicit `None` return at line 36; error in neighboring `solution.json`. Examples-and-route yields 913520.1122932937, matching 913520.1123.
- `examples_only_V1/NRM/results_cases/AUTO/OR-018/15b5c36468a8d7a4/solve.py`: optimize at line 70; assignment at line 77; same missing return. Examples-and-route yields 782819438.64, matching 782819438.6.
- `examples_only_V1/RA/results_cases/AUTO/OR-036/a182e169f1cc159d/solve.py`: optimize at line 31; assignment at line 38; same missing return. Examples-and-route yields 509925, matching reference 509925.
- `examples_only_V1/RA/results_cases/AUTO/OR-043/6ba5f20b37b628ec/solve.py`: defines a function returning `m` at line 40, but never calls it; runner finds no global `m`/`model` and records `KeyError: 'model'`.
- OR-011 has `ModelOutputTruncated` in the examples-only `solution.json`; the alternative route matches 3552260.54.
- OR-029 changes from NRM to RA and still solves, but objective changes from correct 5090631.58 to incorrect 4982034.08.
- OR-080 changes from Others to TP and still solves, but objective changes from correct 14300 to incorrect 15590.

## Interpretation

The GPT-4.1 aggregate improvement of 13 matched problems is strongly driven by reduced execution/interface failures. RA alone rises from ten to 21 objective matches, accounting for eleven of the thirteen net gains. The examples-only RA failures include nine missing-return errors, one undefined global model, one unsupported keyword and one truncated output. This cannot establish that disabling the same-type route improves mathematical modeling. It also does not establish a contradiction with another model that has fewer of these interface failures to recover from.

Full tables, manifests, paired-case transitions, and static function analyses are in `gpt41_audit.json` and the adjacent JSON table files.
