See [SUMMARY.md](SUMMARY.md) for final source provenance, implementation limitations and launch controls.

# ReAct architecture evaluation: v7

Status: necessary full-model gate passed.
606 enabled: False; executed: False.

NRM, RA, TP, AP and FLP use ZERO_SHOT_REACT_DESCRIPTION with CSVQA. Others with CSV retains its two-chain schema workflow; query-only ORLM_QA remains.
The agent chooses CSVQA and receives its Observation before its Final Answer.
The agent must execute CSVQA at least once; multiple calls for different CSV files are allowed. Missing calls, malformed ReAct output and an incomplete Final Answer restart only the agent within the common case deadline, as explicitly requested by the user. Intermediate protocol attempts are logged and are not scored as separate failed cases. The first protocol-valid completion is retained; objective mismatches and generated-code errors never trigger retries.
Few-shot Only uses the same ReAct agent class with an already supplied complete Python
Observation and no data tools.

All failed/error/truncated/timeout cases stay in the prescribed denominators.
Objective Match requires a finite objective from an optimal returned Gurobi model,
with unchanged rel_tol=abs_tol=1e-4. Classification and successful solve are separate.
A partial run is marked incomplete and is not a complete benchmark result.

## Overall comparison under the confirmed policy

Historical policy applies to this preserved version.

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
| NRM | 25 | 25 | 25 | 25 | 25 |
| RA | 22 | 22 | 22 | 22 | 22 |
| TP | 9 | 9 | 9 | 9 | 9 |
| Others | 8 | 8 | 7 | 6 | 5 |
| Mixture | 18 | 18 | 12 | 16 | 15 |

## Full datasets and every sheet

| dataset | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| automatic | 101 | 101 | 94 | 96 | 94 |
| variants | 36 | 36 | 32 | 34 | 34 |
| columns/50pct-S1 | 35 | 35 | 33 | 33 | 33 |
| columns/50pct-S2 | 35 | 35 | 33 | 33 | 33 |
| columns/50pct-S3 | 35 | 35 | 33 | 33 | 33 |
| columns/100pct-S1 | 35 | 35 | 33 | 32 | 32 |
| columns/100pct-S2 | 35 | 35 | 33 | 32 | 30 |
| columns/100pct-S3 | 35 | 35 | 33 | 33 | 32 |
| columns/200pct-S1 | 35 | 35 | 33 | 32 | 32 |
| columns/200pct-S2 | 35 | 35 | 33 | 33 | 33 |
| columns/200pct-S3 | 35 | 35 | 33 | 33 | 31 |

## Ablations and LOTO against this ReAct baseline

| method | version | recorded | classification_correct | solved | objective_match | delta_matches_vs_full | delta_percentage_points | status |
|---|---|---|---|---|---|---|---|---|
| full | v7 | 101 | 94 | 96 | 94 | 0 | 0.0 | complete |
| rag_only | v7 | 101 | 94 | 90 | 86 | -8 | -7.920792079207921 | complete |
| few_shot_only | v7 | 101 | 94 | 75 | 63 | -31 | -30.693069306930692 | complete |
| examples_only |  | 0 |  |  |  |  |  | not_run |
| examples_and_route | v7 | 101 | 0 | 91 | 88 | -6 | -5.9405940594059405 | complete |


Ablations reuse only the newly validated ReAct full classification cache. LOTO
reclassifies after target-reference removal. Examples Only retains every route;
Examples And Route bans the target workflow and checks availability before modeling.
The original semantic-label accuracy is zero by design in the route-removal experiment.
Only fold grouping/exclusion and the scorer use the true class; no gold model/answer
is used to choose a replacement route or generate code.

## Failures

| method | failure_stage | count |
|---|---|---|
| examples_and_route | execution | 9 |
| examples_and_route | mathematical_modeling | 1 |
| examples_and_route | mathematical_modeling_or_code_generation | 3 |
| few_shot_only | code_generation | 14 |
| few_shot_only | data_transfer | 1 |
| few_shot_only | execution | 10 |
| few_shot_only | mathematical_modeling | 1 |
| few_shot_only | mathematical_modeling_or_code_generation | 12 |
| full | code_generation | 20 |
| full | data_extraction | 1 |
| full | data_extraction_or_code_generation | 4 |
| full | execution | 2 |
| full | mathematical_modeling | 1 |
| full | mathematical_modeling_or_code_generation | 7 |
| rag_only | code_generation | 1 |
| rag_only | execution | 1 |
| rag_only | mathematical_modeling | 9 |
| rag_only | mathematical_modeling_or_code_generation | 4 |

| method | dataset | problem_id | failure_stage | failure_type | cause_certainty |
|---|---|---|---|---|---|
| full | automatic | OR-050 | code_generation | source_index_read_as_column | confirmed from saved formulation, source payload and generated program |
| full | automatic | OR-062 | data_extraction | matrix_axis_mapping_unresolved | confirmed from saved CSVQA validation, source payload and generated program |
| full | automatic | OR-074 | code_generation | post_solve_lookup_of_absent_variable_name | confirmed from generated program and traceback |
| full | automatic | OR-084 | execution | external_case_deadline_exceeded | confirmed; no optimal result within the common execution deadline |
| full | automatic | OR-087 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | automatic | OR-097 | execution | TypeError | confirmed; upstream semantic cause may be unresolved |
| full | automatic | OR-100 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/100pct-S1 | 100pct-S1/OR-018 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/100pct-S1 | 100pct-S1/OR-019 | code_generation | invalid_namedtuple_record_access | confirmed by generated program and AttributeError |
| full | columns/100pct-S1 | 100pct-S1/OR-023 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/100pct-S2 | 100pct-S2/OR-009 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/100pct-S2 | 100pct-S2/OR-021 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/100pct-S2 | 100pct-S2/OR-023 | data_extraction_or_code_generation | unresolved_matrix_identifier_mapping | confirmed source identifier mismatch; a justified mapping remains unresolved |
| full | columns/100pct-S2 | 100pct-S2/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/100pct-S2 | 100pct-S2/OR-029 | code_generation | source_index_read_as_column | confirmed by source payload and generated program |
| full | columns/100pct-S3 | 100pct-S3/OR-023 | data_extraction_or_code_generation | unresolved_matrix_identifier_mapping | confirmed source identifier mismatch; a justified mapping remains unresolved |
| full | columns/100pct-S3 | 100pct-S3/OR-025 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/100pct-S3 | 100pct-S3/OR-026 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/200pct-S1 | 200pct-S1/OR-023 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/200pct-S1 | 200pct-S1/OR-026 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/200pct-S1 | 200pct-S1/OR-029 | code_generation | source_index_read_as_column | confirmed by source payload and generated program |
| full | columns/200pct-S2 | 200pct-S2/OR-023 | data_extraction_or_code_generation | unresolved_matrix_identifier_mapping | confirmed source identifier mismatch; a justified mapping remains unresolved |
| full | columns/200pct-S2 | 200pct-S2/OR-026 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/200pct-S3 | 200pct-S3/OR-023 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/200pct-S3 | 200pct-S3/OR-026 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/200pct-S3 | 200pct-S3/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/200pct-S3 | 200pct-S3/OR-029 | code_generation | source_index_read_as_column | confirmed by source payload and generated program |
| full | columns/50pct-S1 | 50pct-S1/OR-021 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/50pct-S1 | 50pct-S1/OR-028 | data_extraction_or_code_generation | unresolved_matrix_identifier_mapping | confirmed source identifier mismatch; a justified mapping remains unresolved |
| full | columns/50pct-S2 | 50pct-S2/OR-023 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/50pct-S2 | 50pct-S2/OR-029 | code_generation | source_index_read_as_column | confirmed by source payload and generated program |
| full | columns/50pct-S3 | 50pct-S3/OR-021 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | columns/50pct-S3 | 50pct-S3/OR-026 | code_generation | invalid_source_record_access | confirmed by saved program and execution exception |
| full | variants | Variant1 | mathematical_modeling | model_output_truncated | confirmed by saved callback and execution record |
| full | variants | Variant16 | code_generation | post_solve_lookup_of_absent_variable_name | confirmed from generated program and traceback |
| rag_only | automatic | OR-007 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-009 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-010 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-011 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-027 | code_generation | none_passed_as_gurobi_numeric_bound | confirmed from generated program and execution exception |
| rag_only | automatic | OR-033 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-039 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| rag_only | automatic | OR-043 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-046 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-047 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-050 | execution | KeyError | confirmed; upstream semantic cause may be unresolved |
| rag_only | automatic | OR-062 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| rag_only | automatic | OR-079 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| rag_only | automatic | OR-099 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| rag_only | automatic | OR-100 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-005 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-013 | data_transfer | complete_source_exceeds_model_context | confirmed by API rejection |
| few_shot_only | automatic | OR-015 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-017 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-023 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-027 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-030 | code_generation | KeyError | confirmed by execution |
| few_shot_only | automatic | OR-031 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| few_shot_only | automatic | OR-035 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-037 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-039 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-046 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-048 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-051 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-052 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-053 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-054 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-057 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-058 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-060 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-061 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-062 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-064 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-065 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-066 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-071 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| few_shot_only | automatic | OR-072 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-073 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-076 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-083 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-084 | execution | external_case_deadline_exceeded | confirmed; no optimal result within the common execution deadline |
| few_shot_only | automatic | OR-085 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-086 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-087 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-097 | code_generation | none_passed_as_gurobi_numeric_bound | confirmed from generated program and execution exception |
| few_shot_only | automatic | OR-099 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| few_shot_only | automatic | OR-100 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| few_shot_only | automatic | OR-101 | code_generation | symbolic_model_transfer_heading_filter_rejection | confirmed from saved formulation and unchanged v7 transfer-filter source |
| examples_and_route | automatic | OR-005 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-021 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-022 | execution | KeyError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-041 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| examples_and_route | automatic | OR-062 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-080 | execution | RuntimeError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-083 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-085 | execution | AttributeError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-086 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-087 | mathematical_modeling | model_output_truncated | confirmed by finish_reason=length callback and saved failure record |
| examples_and_route | automatic | OR-092 | execution | KeyError | confirmed; upstream semantic cause may be unresolved |
| examples_and_route | automatic | OR-097 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| examples_and_route | automatic | OR-100 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |


Unresolved objective mismatches remain unresolved until the saved model/program/source
records support an explanation. Missing CSVQA evidence is explicitly marked, and
fallback totals are lower bounds when evidence is unavailable. No trace is fabricated.

## Repairs, retries and fallbacks

| method | dataset | repair_count | pipeline_retry_count | protocol_retry_count | http_retry_count | recorded_full_data_fallback_count | csvqa_evidence_unavailable_cases |
|---|---|---|---|---|---|---|---|
| full | automatic | 0 | 0 | 0 | 0 | 21 | 0 |
| full | variants | 0 | 0 | 0 | 0 | 5 | 1 |
| full | columns/50pct-S1 | 0 | 0 | 0 | 0 | 9 | 0 |
| full | columns/50pct-S2 | 0 | 0 | 0 | 0 | 9 | 0 |
| full | columns/50pct-S3 | 0 | 0 | 0 | 0 | 11 | 0 |
| full | columns/100pct-S1 | 0 | 0 | 0 | 0 | 12 | 0 |
| full | columns/100pct-S2 | 0 | 0 | 0 | 0 | 11 | 0 |
| full | columns/100pct-S3 | 0 | 0 | 0 | 0 | 12 | 0 |
| full | columns/200pct-S1 | 0 | 0 | 0 | 0 | 10 | 0 |
| full | columns/200pct-S2 | 0 | 0 | 0 | 0 | 10 | 0 |
| full | columns/200pct-S3 | 0 | 0 | 0 | 0 | 7 | 0 |
| rag_only | automatic | 0 | 0 | 0 | 0 | 21 | 0 |
| few_shot_only | automatic | 0 | 0 | 0 | 0 | 0 | 0 |
| examples_and_route | automatic | 0 | 0 | 0 | 0 | 7 | 0 |


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
python scripts/evaluate_react_revision_20261006.py --method rag_only --version v7 --workers 3 --baseline-study --run
python scripts/evaluate_react_revision_20261006.py --method few_shot_only --version v7 --workers 3 --baseline-study --run
python scripts/evaluate_react_revision_20261006.py --method examples_and_route --version v7 --workers 3 --baseline-study --run
python scripts/report_react_revision_20261006.py --version v7 --output-dir outputs/react_revision_20261006/report_v7_reproduced_summary
```

Every case/version has one scored pipeline attempt. This version's protocol restarts
are logged within that attempt. Resuming preserves failures as well as successes.
Changed code or data requires a fresh result directory. Every manifest freezes the
notebook, input hashes, fixed model snapshot and execution settings. Source files,
model/code artifacts and logs are retained. 606 stays disabled until this revision
passes the required full-model gate and all nine sheets have been reviewed.
