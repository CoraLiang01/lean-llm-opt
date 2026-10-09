# ReAct architecture evaluation: v13

This report scores the frozen v13 source. All 452 prescribed outcomes are recorded.
The coordinator complete flag remains false after API credit_balance_exhausted;
three API failures remain in the denominator. The current delivered files are
v14 with zero inference cases and unverified performance. [Latest status](../STATS_LATEST.md).

Status: overall launch gate not passed.
606 enabled: False; executed: False.

All six CSV workflows use ZERO_SHOT_REACT_DESCRIPTION with required CSVQA. Others/Mixture use the same symbolic-model and code-generation boundary. The original query-only ORLM_QA route remains.
The agent chooses CSVQA and receives its Observation before its Final Answer.
The agent must execute CSVQA at least once; multiple calls for different CSV files are allowed. Missing calls, malformed ReAct output and an incomplete Final Answer restart only the agent within the common case deadline, as explicitly requested by the user. Intermediate protocol attempts are logged and are not scored as separate failed cases. The first protocol-valid completion is retained; objective mismatches and generated-code errors never trigger retries.
Few-shot Only uses the same ReAct agent class with an already supplied complete Python
Observation and no data tools.

All failed/error/truncated/timeout cases stay in the prescribed denominators.
Objective Match requires a finite objective from an optimal returned Gurobi model,
with unchanged rel_tol=abs_tol=1e-4. Classification and successful solve are separate.
A partial run is marked incomplete and is not a complete benchmark result.

## Overall comparison under the confirmed policy

| group | expected | baseline_match | current_match | delta_matches | baseline_accuracy_percent | current_accuracy_percent |
|---|---|---|---|---|---|---|
| 101 | 101 | 94 | 92 | -2 | 93.06930693069307 | 91.08910891089108 |
| Variants | 36 | 34 | 30 | -4 | 94.44444444444444 | 83.33333333333333 |
| Redundant columns (all nine sheets) | 315 | 289 | 295 | 6 | 91.74603174603175 | 93.65079365079364 |

Equal mean of the three group accuracies: baseline 93.0866%; current 89.3577%; change -3.7288 percentage points. Pooled matches: 417/452 to 417/452.
The current launch policy permits individual dataset, sheet and class declines.
It still requires 101 >= 93/101, AP/FLP/NRM/RA/TP >= 92% each, Variants >= 32/36,
unchanged inputs and complete evaluation of all nine redundancy sheets. Redundancy
targets remain improvement goals. All declines are reported. Partial totals do not
establish improvement. New studies are blocked until the complete launch gate passes.

## Full 101 by class

| class | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| AP | 5 | 5 | 5 | 5 | 5 |
| FLP | 14 | 14 | 14 | 13 | 13 |
| NRM | 25 | 25 | 25 | 24 | 24 |
| RA | 22 | 22 | 22 | 22 | 22 |
| TP | 9 | 9 | 9 | 9 | 9 |
| Others | 8 | 8 | 7 | 8 | 7 |
| Mixture | 18 | 18 | 13 | 14 | 12 |

## Full datasets and every sheet

| dataset | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| automatic | 101 | 101 | 95 | 95 | 92 |
| variants | 36 | 36 | 30 | 30 | 30 |
| columns/50pct-S1 | 35 | 35 | 32 | 34 | 33 |
| columns/50pct-S2 | 35 | 35 | 33 | 35 | 35 |
| columns/50pct-S3 | 35 | 35 | 33 | 34 | 33 |
| columns/100pct-S1 | 35 | 35 | 33 | 35 | 34 |
| columns/100pct-S2 | 35 | 35 | 33 | 34 | 34 |
| columns/100pct-S3 | 35 | 35 | 33 | 32 | 30 |
| columns/200pct-S1 | 35 | 35 | 33 | 34 | 33 |
| columns/200pct-S2 | 35 | 35 | 33 | 34 | 33 |
| columns/200pct-S3 | 35 | 35 | 31 | 32 | 30 |

## Ablations and LOTO against this ReAct baseline

| method | version | recorded | classification_correct | solved | objective_match | delta_matches_vs_full | delta_percentage_points | status |
|---|---|---|---|---|---|---|---|---|
| full | v13 | 101 | 95 | 95 | 92 | 0 | 0.0 | complete |
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
| full | classification | 2 |
| full | classification_and_mathematical_modeling | 1 |
| full | code_generation | 5 |
| full | data_extraction | 7 |
| full | mathematical_modeling | 16 |
| full | mathematical_modeling_or_code_generation | 4 |

| method | dataset | problem_id | failure_stage | failure_type | cause_certainty |
|---|---|---|---|---|---|
| full | automatic | OR-028 | code_generation | unsupported_None_upper_bound | confirmed from generated addVars call and execution TypeError |
| full | automatic | OR-062 | data_extraction | missing_trusted_customer_city_mapping | confirmed incompatible source identifiers; intended mapping unresolved |
| full | automatic | OR-080 | mathematical_modeling | extra_minimum_down_time_period | confirmed against original query and saved model/code; isolated objective impact untested |
| full | automatic | OR-086 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | automatic | OR-087 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | automatic | OR-090 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | automatic | OR-092 | mathematical_modeling | per_truck_capacity_applied_to_total_route_cargo | confirmed from original query, saved model and generated variable bound |
| full | automatic | OR-097 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | automatic | OR-100 | classification_and_mathematical_modeling | route_disagreement_and_ambiguous_integrality | classification disagreement confirmed; production integrality interpretation unresolved |
| full | columns/100pct-S1 | 100pct-S1/OR-023 | mathematical_modeling_or_code_generation | unsupported_positional_customer_city_mapping | positional assumption confirmed from saved code; correct crosswalk unresolved |
| full | columns/100pct-S2 | 100pct-S2/OR-023 | data_extraction | missing_trusted_customer_city_mapping | confirmed source identifier mismatch; intended mapping unresolved |
| full | columns/100pct-S3 | 100pct-S3/OR-008 | code_generation | unsupported_None_upper_bound | confirmed addVars call and execution TypeError |
| full | columns/100pct-S3 | 100pct-S3/OR-023 | data_extraction | missing_trusted_customer_city_mapping | confirmed source identifier mismatch; intended mapping unresolved |
| full | columns/100pct-S3 | 100pct-S3/OR-025 | mathematical_modeling | missing_activity_activation_link | confirmed saved model and code; conflicting common instruction also identified |
| full | columns/100pct-S3 | 100pct-S3/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch_with_matrix_axis_assumption | axis assignment confirmed in code; intended orientation unresolved without explicit source distinction |
| full | columns/100pct-S3 | 100pct-S3/OR-035 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | columns/200pct-S1 | 200pct-S1/OR-023 | data_extraction | missing_trusted_customer_city_mapping | confirmed source identifier mismatch; intended mapping unresolved |
| full | columns/200pct-S1 | 200pct-S1/OR-025 | mathematical_modeling | missing_activity_activation_link | confirmed saved model and code; conflicting common instruction also identified |
| full | columns/200pct-S2 | 200pct-S2/OR-023 | data_extraction | missing_trusted_customer_city_mapping | confirmed source identifier mismatch; intended mapping unresolved |
| full | columns/200pct-S2 | 200pct-S2/OR-025 | mathematical_modeling | missing_activity_activation_link | confirmed saved model and code; conflicting common instruction also identified |
| full | columns/200pct-S3 | 200pct-S3/OR-023 | code_generation | fabricated_missing_coefficients | confirmed saved generated program; missing source crosswalk remains unresolved |
| full | columns/200pct-S3 | 200pct-S3/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/200pct-S3 | 200pct-S3/OR-029 | code_generation | api_credit_balance_exhausted | confirmed API 429 insufficient_quota/credit_balance_exhausted response |
| full | columns/200pct-S3 | 200pct-S3/OR-034 | classification | api_credit_balance_exhausted | confirmed API 429 insufficient_quota/credit_balance_exhausted response |
| full | columns/200pct-S3 | 200pct-S3/OR-035 | classification | api_credit_balance_exhausted | confirmed API 429 insufficient_quota/credit_balance_exhausted response |
| full | columns/50pct-S1 | 50pct-S1/OR-023 | data_extraction | missing_trusted_customer_city_mapping | confirmed source identifier mismatch; intended mapping unresolved |
| full | columns/50pct-S1 | 50pct-S1/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch_with_matrix_axis_assumption | axis assignment confirmed in code; intended orientation unresolved without explicit source distinction |
| full | columns/50pct-S3 | 50pct-S3/OR-009 | mathematical_modeling | ambiguous_development_quantity_domain | continuous domain confirmed; intended integrality unresolved from source question |
| full | columns/50pct-S3 | 50pct-S3/OR-023 | data_extraction | missing_trusted_customer_city_mapping | confirmed source identifier mismatch; intended mapping unresolved |
| full | variants | Variant7 | mathematical_modeling | hub_throughput_double_counted | confirmed query/model/code discrepancy; infeasibility reported by solver |
| full | variants | Variant13 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | variants | Variant14 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | variants | Variant20 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | variants | Variant21 | mathematical_modeling | output_truncated_after_valid_observation | confirmed saved API completion and CSVQA trace |
| full | variants | Variant32 | code_generation | retransmission_duplicates_in_variable_keys | confirmed saved select_latest/addVars implementation and execution KeyError |


Unresolved objective mismatches remain unresolved until the saved model/program/source
records support an explanation. Missing CSVQA evidence is explicitly marked, and
fallback totals are lower bounds when evidence is unavailable. No trace is fabricated.

## Repairs, retries and fallbacks

| method | dataset | repair_count | pipeline_retry_count | protocol_retry_count | http_retry_count | recorded_full_data_fallback_count | csvqa_evidence_unavailable_cases |
|---|---|---|---|---|---|---|---|
| full | automatic | 0 | 0 | 0 | 0 | 12 | 0 |
| full | variants | 0 | 0 | 0 | 0 | 4 | 0 |
| full | columns/50pct-S1 | 0 | 0 | 0 | 0 | 14 | 0 |
| full | columns/50pct-S2 | 0 | 0 | 0 | 0 | 6 | 0 |
| full | columns/50pct-S3 | 0 | 0 | 0 | 0 | 7 | 0 |
| full | columns/100pct-S1 | 0 | 0 | 0 | 0 | 5 | 0 |
| full | columns/100pct-S2 | 0 | 0 | 0 | 0 | 10 | 0 |
| full | columns/100pct-S3 | 0 | 0 | 0 | 0 | 5 | 0 |
| full | columns/200pct-S1 | 0 | 0 | 0 | 0 | 10 | 0 |
| full | columns/200pct-S2 | 0 | 0 | 0 | 0 | 6 | 0 |
| full | columns/200pct-S3 | 0 | 0 | 0 | 9 | 6 | 0 |


HTTP retries retain the original SDK transport policy and are recorded separately.
ReAct protocol restarts follow this version's recorded user policy. Code and solver
repair and objective-based retries remain disabled. Each worker uses Gurobi Threads=2;
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
python scripts/evaluate_react_revision_20261006.py --method full --version v13 --workers 6 --run
python scripts/evaluate_react_revision_20261006.py --method rag_only --version v13 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method few_shot_only --version v13 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_only --version v13 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_and_route --version v13 --workers 3 --run
python scripts/report_react_revision_20261006.py --version v13
```

Every case/version has one scored pipeline attempt. This version's protocol restarts
are logged within that attempt. Resuming preserves failures as well as successes.
Changed code or data requires a fresh result directory. Every manifest freezes the
notebook, input hashes, fixed model snapshot and execution settings. Source files,
model/code artifacts and logs are retained. 606 stays disabled until this revision
passes the required full-model gate and all nine sheets have been reviewed.
