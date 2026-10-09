# Latest recorded results: evaluated v13; current v14 unmeasured

All 452 prescribed v13 outcomes have been recorded once. Errors, truncations,
non-optimal results and objective mismatches remain in every denominator.
The coordinator's complete flag is false after three final API credit-balance errors;
no case is missing or excluded. Matching tolerances and inputs are unchanged.
These scores describe frozen v13 SHA256
`164fa23f8d08ffb34e7b32d12bb298eea548d538279acadbf6fca1acc9c2db78`.
The current files are v14, with zero inference cases and unverified performance.

## Three data groups

| Group | Classification correct | Optimal solve | Objective Match | Objective Match accuracy |
|---|---|---|---|---|
| 101 | 95/101 | 95/101 | 92/101 | 91.09% |
| Variants | 30/36 | 30/36 | 30/36 | 83.33% |
| Redundant columns | 294/315 | 304/315 | 295/315 | 93.65% |

The user-confirmed equal mean of these three accuracies is
**89.3577%**, versus ReAct v7
**93.0866%**:
**-3.7288 percentage points**.
Pooled Objective Match is **417/452 (92.2566%)** in both passes. Pooled equality
does not satisfy the user-confirmed equal-group improvement criterion.
Across corresponding cases, 21 improved and 21 regressed; 410 were unchanged.

## 101 by category

| Category | Classification correct | Optimal solve | Objective Match |
|---|---|---|---|
| AP | 5/5 | 5/5 | 5/5 |
| FLP | 14/14 | 13/14 | 13/14 |
| NRM | 25/25 | 24/25 | 24/25 |
| RA | 22/22 | 22/22 | 22/22 |
| TP | 9/9 | 9/9 | 9/9 |
| Others | 7/8 | 8/8 | 7/8 |
| Mixture | 13/18 | 14/18 | 12/18 |

Overall: classification **95/101**, optimal solve **95/101**, Objective Match **92/101**.
All five main categories exceed the necessary 92% floor. FLP 13/14 is below the
preferred 95% target. The 101 total is below 93/101 and Variants is below 32/36.

## All nine redundancy sheets

| Sheet | Classification correct | Optimal solve | Objective Match | Non-matching case IDs |
|---|---|---|---|---|
| 50pct-S1 | 32/35 | 34/35 | 33/35 | OR-023, OR-028 |
| 50pct-S2 | 33/35 | 35/35 | 35/35 | None |
| 50pct-S3 | 33/35 | 34/35 | 33/35 | OR-009, OR-023 |
| 100pct-S1 | 33/35 | 35/35 | 34/35 | OR-023 |
| 100pct-S2 | 33/35 | 34/35 | 34/35 | OR-023 |
| 100pct-S3 | 33/35 | 32/35 | 30/35 | OR-008, OR-023, OR-025, OR-028, OR-035 |
| 200pct-S1 | 33/35 | 34/35 | 33/35 | OR-023, OR-025 |
| 200pct-S2 | 33/35 | 34/35 | 33/35 | OR-023, OR-025 |
| 200pct-S3 | 31/35 | 32/35 | 30/35 | OR-023, OR-028, OR-029, OR-034, OR-035 |

100pct-S3 and 200pct-S3 are below both 31/35 and 90%; further improvement is needed.
200pct-S3 includes three API-credit failures. They remain scored as failures.
The customer-to-city alignment remains unsupported in the sources; successful
numerical matches using positional assumptions are separately flagged.

## Failed 101 and Variants cases

101: OR-028, OR-062, OR-080, OR-086, OR-087, OR-090, OR-092, OR-097, OR-100.
Variants: 7, 13, 14, 20, 21, 32.
Confirmed failures include unsupported None bounds, missing source ID mappings,
excess minimum downtime, an incorrect aggregate truck-cargo bound, doubled hub
throughput, missing activation links, duplicate retransmission keys and truncation.
Ambiguous domains and matrix-axis interpretations remain explicitly unresolved.
[Every failed case and its evidence](report_v13/failures.csv).

## Tool use, retries, fallbacks and studies

CSVQA attempted and returned a valid Observation for **450/452** cases, once each.
Two credit failures occurred before classification/CSVQA; one occurred at code
generation after CSVQA. There were **0** protocol restarts, **0** pipeline retries,
**0** repairs, **9** original SDK HTTP retries and **85** full-source Observation
fallbacks. A fallback is not a pipeline rerun and does not supply a missing crosswalk.
There were **9** formulation truncations, all after valid CSVQA Observations.

Current ablation and LOTO notebooks share v14 and are offline checked; no new study
has been launched because the full gate fails and API credits are exhausted.
606 remains off and unexecuted. Historical direct-call scores are separate and are
not reused as current ReAct results. All five direct-call files are unchanged.

## Artifacts and current source

- [Detailed evaluated-source report](report_v13/REPORT.md)
- [All 452 case outcomes](report_v13/all_case_results.csv)
- [Case transitions versus v7](report_v13/full_vs_v7_cases.csv)
- [Current v14 change and implementation review](CODE_REVIEW_V14.md)
- [Frozen v13 source](full_v13/frozen_notebook.ipynb)
- [v13 inputs and configuration](full_v13/manifest.json)
- [Current v14 input/configuration preflight](full_v14/manifest.json)

v14 removes one contradictory common prompt statement; its benefit is unverified.
No repair, outcome retry, reference modification or relaxed tolerance is introduced.
The 1,133 input hashes are unchanged. One pass does not establish stability.
