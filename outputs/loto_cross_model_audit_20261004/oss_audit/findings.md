# OSS20b LOTO saved-output audit (2026-10-04)

Source directory: `/Users/cora/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_yj16fcohm45a22_43e6/msg/file/2026-10/LOTO_outputs`.

Audit reads existing records and generated artifacts. It does not execute generated optimization code, call models, modify notebooks, or modify source results. Counts are verified saved-run results, not newly rerun results.

## Verified totals

| Variant | Cases | Solved | Objective matched | Solved but mismatched | Failures | Classification correct |
|---|---:|---:|---:|---:|---:|---:|
| Examples only | 101 | 94 | 88 | 6 | 7 | 84 |
| Examples and route | 101 | 93 | 85 | 8 | 8 | N/A |

Every variant has 101 unique IDs, complete seven-fold manifest coverage, `cache_source=computed` for all 101 rows, and consistent fold CSV / fold report / overall report / solution JSON records. Stored objective matching agrees with recomputation using `math.isclose(rel_tol=1e-4, abs_tol=1e-4)`. All solved flags agree with saved `OPTIMAL` solver status, and all solved objective values agree with solution JSON.

All fourteen manifests identify `gpt-oss:20b`, baseline SHA256 `912c7d1beca1f99cf05eb681662ea85cab61850428a4479d79577eec1df8d795`, and reference SHA256 `6b2987f96aeacece9f458d55aca072436922aec963be16223ac30a8890f79097`. This identifies recorded configuration, not the immutable model digest or complete runtime environment. All same-type reference removals match after normalizing `UFLP` to `FLP`.

No data-integrity inconsistencies were found. Experiment 2 has two forbidden classification attempts, both correctly blocked before modeling/solving; these are experiment failures, not executed forbidden routes. It also has one missing classification due to an Ollama/CUDA failure.

## By-type results

| True type | Cases | E1 solved | E1 matched | E2 solved | E2 matched | Match change |
|---|---:|---:|---:|---:|---:|---:|
| TP | 9 | 8 | 8 | 7 | 7 | -1 |
| NRM | 25 | 25 | 25 | 25 | 25 | 0 |
| RA | 22 | 22 | 20 | 21 | 20 | 0 |
| FLP | 14 | 12 | 12 | 13 | 13 | +1 |
| AP | 5 | 5 | 5 | 5 | 5 | 0 |
| Mixture | 18 | 15 | 12 | 15 | 11 | -1 |
| Others | 8 | 7 | 6 | 7 | 4 | -2 |

Thus OSS20b has no NRM/RA execution-recovery advantage available: E1 already solves all 47 NRM/RA problems and matches 45/47. E2 matches the same 45/47, despite individual RA gains and losses. This sharply differs from GPT4.1 V1's many execution failures in these families.

## Paired decomposition

- Both experiments match: 80 cases.
- E2 gains: 5 cases (3 recovered execution failures, 2 corrected mismatches).
- E2 losses: 8 cases (5 new failures, 3 new mismatches).
- Neither matches: 8 cases.
- Net: 5 - 8 = -3 objective matches.
- Both solve: 88 cases; E1 matches 83/88 (94.32%), E2 matches 82/88 (93.18%).
- E1-only solved: 6 cases. E2-only solved: 5 cases. Neither solved: 2 cases.

| Case | Type | E1 route | E2 selection | Change and concrete evidence |
|---|---|---|---|---|
| OR-009 | TP | TP | Others | Gain from failure: E1 cannot convert string `S1` to float. E2 objective 238781.51192549663 matches 238781.5119. |
| OR-039 | RA | RA | Others | Gain from mismatch: 2502500 to 429000. E1 capacity constraint omits product Weight; E2 uses weighted capacity. |
| OR-040 | RA | RA | Others | Gain from mismatch: 4198040 to 74260. E1 again uses sum(x) <= capacity; E2 uses sum(weight*x) <= capacity. |
| OR-062 | FLP | FLP | Others | Gain from failure: E1 `KeyError: Customer_1`; E2 matches 172612.46. |
| OR-076 | FLP | FLP | Others | Gain from failure: E1 out-of-bounds column index20 for width20; E2 matches 103090. |
| OR-005 | TP | TP | Missing | Loss: E2 classification stage Ollama500 / CUDA illegal memory access, before any modeling or solving. |
| OR-007 | TP | TP | TP, blocked | Loss: E2 returns unavailable TP; `DisabledRouteError`. |
| OR-041 | RA | RA | Others | Loss: E2 uses continuous variables (4408.523076923077), E1 uses integer variables (3912). Query does not explicitly state integer, so treat this as domain-assumption/benchmark mismatch rather than unambiguously proven prompt violation. |
| OR-057 | RA | RA | RA, blocked | Loss: E2 returns unavailable RA; `DisabledRouteError`. |
| OR-059 | FLP | FLP | Others | Loss: E2 `AttributeError: str object has no attribute astype`. |
| OR-087 | Mixture | Others | RA | Loss: E2 reads demand, price, cost, quota all from records[0] of the same table; price equals cost, positive fixed costs => zero production/objective -0.0 rather than 6243055.96. Also adds a shared 22-day constraint absent in E1. |
| OR-072 | Others | Others | RA | Loss: E2 `KeyError: 1`. |
| OR-082 | Others | Others | TP | Loss: E2 reconstructs distances by iterating every record and every origin, overwriting every origin row with current record; final record wins. Objective231 vs199. Mathematical model text is a reasonable TSP; generated data-binding code corrupts coefficients. |

## Classification matrix: examples only

Rows = true types; columns = predicted semantic labels, including predictions preceding execution failures.

| True | TP | NRM | RA | FLP | AP | Mixture | Others | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NRM | 0 | 23 | 2 | 0 | 0 | 0 | 0 | 0 |
| RA | 0 | 0 | 21 | 0 | 0 | 0 | 1 | 0 |
| FLP | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 0 |
| AP | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 |
| Mixture | 1 | 0 | 8 | 0 | 1 | 5 | 3 | 0 |
| Others | 0 | 0 | 1 | 0 | 0 | 0 | 7 | 0 |

## Classification matrix: examples and route

The two diagonal entries are forbidden outputs blocked at route selection, not executed workflows. Missing = OR-005 classification service failure.

| True | TP | NRM | RA | FLP | AP | Mixture | Others | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP | 1 | 0 | 0 | 0 | 0 | 0 | 7 | 1 |
| NRM | 0 | 0 | 24 | 0 | 0 | 0 | 1 | 0 |
| RA | 0 | 0 | 1 | 0 | 0 | 0 | 21 | 0 |
| FLP | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 |
| AP | 3 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| Mixture | 4 | 0 | 13 | 0 | 1 | 0 | 0 | 0 |
| Others | 1 | 0 | 6 | 0 | 1 | 0 | 0 | 0 |

Mixture and Others both map to physical Others workflow; workflow selection matrices are saved separately as CSV.

## Interpretation and limitations

The result is consistent with competing mechanisms: route removal can correct or avoid a bad data/model path, but can introduce service failures, constraint-following failures, variable-domain changes and data-binding errors on alternative workflows. This saved run provides concrete examples of each direction. It does not establish a general model-capacity explanation or a statistically robust ranking from one run.

The CUDA failure is recorded in classification, not in Gurobi. It should be separately reported and, if reruns are authorized, retested under a prespecified infrastructure-retry policy. The two forbidden outputs are genuine measured limitations of the current classifier+guard policy; silently rerouting/retrying them after seeing scores would change the evaluation protocol.

All original results should be retained when code is repaired. A common revised policy would require all four conditions to be evaluated consistently; saved objective values cannot be updated merely by repairing generated code.

## Artifact locations

Relative to the source directory above:

- `examples_and_route/Mixture/results_cases/AUTO/OR-087/solve.py` and `model.md`.
- `examples_and_route/Others/results_cases/AUTO/OR-082/solve.py` and `data_overview.md`.
- `examples_only/RA/results_cases/AUTO/OR-039/solve.py`, paired with `examples_and_route/RA/results_cases/AUTO/OR-039/solve.py`.
- `examples_only/RA/results_cases/AUTO/OR-040/solve.py`, paired with `examples_and_route/RA/results_cases/AUTO/OR-040/solve.py`.
- `examples_and_route/TP/results_cases/AUTO/OR-005/solution.json`.
- `examples_and_route/TP/results_cases/AUTO/OR-007/solution.json`.
- `examples_and_route/RA/results_cases/AUTO/OR-057/solution.json`.

Machine-readable audit outputs: `audit.json`, `manifest_summary.json`, `paired_cases.csv`, two `*_case_details.csv`, two `*_by_type.csv`, and four `*_matrix.csv`. `audit_oss.py` reproduces this read-only analysis without invoking any saved code.
