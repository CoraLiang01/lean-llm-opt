# Evaluation report: v4

Status: necessary full-model gate passed.
606 switch enabled: True; 606 executed: False.
All rates use the complete prespecified dataset size. Failed, truncated, erroneous,
interrupted and unavailable results are never removed from the denominator. Pending
cases are marked incomplete; a partial run is not a complete benchmark result.
Objective Match uses unchanged `rel_tol=1e-4, abs_tol=1e-4` and requires a finite
objective from an optimal live Gurobi model. Classification accuracy compares the
seven semantic labels; solve success requires completed execution/result extraction
and an optimal live Gurobi model. An optimal printed value followed by a program error
is not a valid returned result and remains a failed case.
Neither classification correctness nor solver optimality implies Objective Match.

## Full model: 101 cases by class

| class | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| AP | 5 | 5 | 5 | 5 | 5 |
| FLP | 14 | 14 | 14 | 14 | 14 |
| NRM | 25 | 25 | 25 | 25 | 25 |
| RA | 22 | 22 | 22 | 22 | 22 |
| TP | 9 | 9 | 9 | 9 | 9 |
| Others | 8 | 8 | 7 | 7 | 6 |
| Mixture | 18 | 18 | 12 | 17 | 16 |

## Full model: every dataset and sheet

| dataset | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| automatic | 101 | 101 | 94 | 99 | 97 |
| variants | 36 | 36 | 31 | 34 | 34 |
| columns/50pct-S1 | 35 | 35 | 33 | 34 | 34 |
| columns/50pct-S2 | 35 | 35 | 33 | 35 | 35 |
| columns/50pct-S3 | 35 | 35 | 33 | 34 | 34 |
| columns/100pct-S1 | 35 | 35 | 33 | 34 | 33 |
| columns/100pct-S2 | 35 | 35 | 33 | 33 | 33 |
| columns/100pct-S3 | 35 | 35 | 33 | 35 | 34 |
| columns/200pct-S1 | 35 | 35 | 33 | 34 | 34 |
| columns/200pct-S2 | 35 | 35 | 33 | 35 | 34 |
| columns/200pct-S3 | 35 | 35 | 33 | 34 | 32 |

## Ablations and LOTO compared with the same final full definitions

| method | version | recorded | classification_correct | solved | objective_match | delta_matches_vs_full | delta_percentage_points | status |
|---|---|---|---|---|---|---|---|---|
| full | v4 | 101 | 94 | 99 | 97 | 0 | 0.0 | complete |
| rag_only | v4 | 101 | 94 | 79 | 77 | -20 | -19.801980198019802 | complete |
| few_shot_only | v5 | 101 | 94 | 84 | 81 | -16 | -15.841584158415841 | complete |
| examples_only | v4 | 101 | 91 | 74 | 73 | -24 | -23.762376237623762 | complete |
| examples_and_route | v4 | 101 | 0 | 95 | 91 | -6 | -5.9405940594059405 | complete |

Route-removal LOTO gold-label classification accuracy is expected to be zero:
the held-out label's workflow is deliberately unavailable. Selected-route validity
is a separate constraint, audited before formulation and reported in case records.
The true type is used only for fold grouping/exclusion, never to choose a replacement route.
`experiments_by_class.csv` reports every class/fold; `paired_changes.csv` identifies
each improvement and regression against this exact frozen full baseline.

## Failure evidence

| method | failure_stage | count |
|---|---|---|
| examples_and_route | classification_or_mathematical_modeling | 1 |
| examples_and_route | code_generation | 5 |
| examples_and_route | data_transfer | 1 |
| examples_and_route | mathematical_modeling | 3 |
| examples_only | classification_or_mathematical_modeling | 1 |
| examples_only | code_generation | 1 |
| examples_only | mathematical_modeling | 26 |
| few_shot_only | classification_or_mathematical_modeling | 1 |
| few_shot_only | code_generation | 16 |
| few_shot_only | data_transfer | 1 |
| few_shot_only | execution | 1 |
| few_shot_only | mathematical_modeling | 1 |
| full | classification_or_mathematical_modeling | 1 |
| full | code_generation | 5 |
| full | data_extraction_or_mathematical_modeling | 4 |
| full | execution | 1 |
| full | mathematical_modeling | 3 |
| full | mathematical_modeling_or_code_generation | 3 |
| full | result_extraction | 1 |
| rag_only | classification_or_mathematical_modeling | 1 |
| rag_only | code_generation | 1 |
| rag_only | mathematical_modeling | 22 |


| method | dataset | problem_id | failure_stage | failure_type | cause_certainty |
|---|---|---|---|---|---|
| full | automatic | OR-080 | code_generation | pre_horizon_index_outside_variable_domain | confirmed from generated program and execution traceback |
| full | automatic | OR-084 | execution | external_case_deadline_exceeded | confirmed; no optimal result within the common execution deadline |
| full | automatic | OR-087 | mathematical_modeling | shared_production_capacity_interpretation | shared-capacity constraint confirmed; intended line-capacity interpretation unresolved |
| full | automatic | OR-100 | classification_or_mathematical_modeling | misclassification_and_unstated_integrality | classification error and generated integer domain confirmed; intended domain not explicit |
| full | columns/100pct-S1 | 100pct-S1/OR-023 | mathematical_modeling_or_code_generation | unvalidated_customer_city_mapping | arbitrary mapping confirmed from program; intended source association unresolved |
| full | columns/100pct-S1 | 100pct-S1/OR-033 | result_extraction | generated_post_solve_name_lookup_after_bulk_name_normalization | confirmed from program and traceback |
| full | columns/100pct-S2 | 100pct-S2/OR-017 | code_generation | table_record_payload_levels_confused | confirmed from program and traceback |
| full | columns/100pct-S2 | 100pct-S2/OR-023 | data_extraction_or_mathematical_modeling | unresolved_cross_file_customer_identifiers | confirmed from saved payload, validation and program |
| full | columns/100pct-S3 | 100pct-S3/OR-028 | mathematical_modeling | divisible_transportation_instead_of_single_source_assignment | continuous split-flow model confirmed; intended assignment convention unresolved |
| full | columns/200pct-S1 | 200pct-S1/OR-023 | data_extraction_or_mathematical_modeling | unresolved_cross_file_customer_identifiers | confirmed from saved payload, validation and program |
| full | columns/200pct-S2 | 200pct-S2/OR-023 | mathematical_modeling_or_code_generation | unvalidated_customer_city_mapping | arbitrary mapping confirmed from program; intended source association unresolved |
| full | columns/200pct-S3 | 200pct-S3/OR-013 | code_generation | execution_namespace_confused_with_host_module | confirmed from program and traceback |
| full | columns/200pct-S3 | 200pct-S3/OR-023 | mathematical_modeling_or_code_generation | unvalidated_customer_city_mapping | arbitrary mapping confirmed from program; intended source association unresolved |
| full | columns/200pct-S3 | 200pct-S3/OR-028 | mathematical_modeling | divisible_transportation_instead_of_single_source_assignment | continuous split-flow model confirmed; intended assignment convention unresolved |
| full | columns/50pct-S1 | 50pct-S1/OR-023 | data_extraction_or_mathematical_modeling | unresolved_cross_file_customer_identifiers | confirmed from saved source payload, validation and program |
| full | columns/50pct-S3 | 50pct-S3/OR-023 | data_extraction_or_mathematical_modeling | unresolved_cross_file_customer_identifiers | confirmed from saved source payload, validation and program |
| full | variants | Variant29 | code_generation | decision_variable_container_shadowed_by_loop_identifier | confirmed from generated program and execution traceback |
| full | variants | Variant36 | code_generation | decision_variable_container_shadowed_by_loop_identifier | confirmed from generated program and execution traceback |
| rag_only | automatic | OR-003 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-006 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-007 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-010 | mathematical_modeling | ModelOutputTruncated | confirmed |
| rag_only | automatic | OR-016 | mathematical_modeling | ModelOutputTruncated | confirmed |
| rag_only | automatic | OR-018 | mathematical_modeling | ModelOutputTruncated | confirmed |
| rag_only | automatic | OR-032 | mathematical_modeling | ModelOutputTruncated | confirmed |
| rag_only | automatic | OR-035 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-037 | mathematical_modeling | APIConnectionError | confirmed |
| rag_only | automatic | OR-044 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-045 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-046 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-048 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-052 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-053 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-054 | mathematical_modeling | ModelOutputTruncated | confirmed |
| rag_only | automatic | OR-055 | mathematical_modeling | ModelOutputTruncated | confirmed |
| rag_only | automatic | OR-057 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-060 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-063 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-065 | mathematical_modeling | APITimeoutError | confirmed |
| rag_only | automatic | OR-080 | mathematical_modeling | extra_initial_ramp_and_shutdown_period | confirmed against original query and saved mathematical model/code |
| rag_only | automatic | OR-085 | code_generation | itertuples_invalid_column_attribute | confirmed from generated code and traceback |
| rag_only | automatic | OR-100 | classification_or_mathematical_modeling | misclassification_and_unstated_integrality | classification error and generated integer domain confirmed; intended domain not explicit |
| few_shot_only | automatic | OR-005 | code_generation | unresolved_axis_identifier_mapping | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-013 | data_transfer | context_length_exceeded | confirmed by API rejection; complete input exceeds the fixed model context |
| few_shot_only | automatic | OR-026 | code_generation | nullable_boolean_mask | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-027 | code_generation | empty_exact_match_category_selection | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-030 | code_generation | coefficient_and_quantity_aggregation | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-045 | code_generation | numpy_scalar_rejected | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-048 | code_generation | string_and_numeric_dictionary_keys | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-049 | code_generation | iterrows_numeric_identifier_coercion | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-050 | code_generation | iterrows_numeric_identifier_coercion | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-062 | code_generation | unresolved_customer_axis_mapping | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-072 | code_generation | time_label_parsed_as_integer | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-073 | code_generation | source_parameter_row_caption_mismatch | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-074 | code_generation | half_hour_time_ranges_as_24_integer_hours | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-083 | code_generation | worker_task_axes_transposed | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-084 | execution | external_case_deadline_exceeded | confirmed; no optimal result within the common execution deadline |
| few_shot_only | automatic | OR-085 | code_generation | irrelevant_blank_diagonal_validation | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-087 | mathematical_modeling | shared_production_capacity_interpretation | shared-capacity constraint confirmed; intended line-capacity interpretation unresolved |
| few_shot_only | automatic | OR-095 | code_generation | resource_identifier_spacing_mismatch | confirmed from saved source data/model/program |
| few_shot_only | automatic | OR-099 | code_generation | matrix_axis_identifier_mismatch | confirmed generated source-axis lookup mismatch |
| few_shot_only | automatic | OR-100 | classification_or_mathematical_modeling | misclassification_and_unstated_integrality | classification error and generated integer domain confirmed; intended domain not explicit |
| examples_only | automatic | OR-003 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-006 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-007 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-009 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-010 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-011 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-013 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-015 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-032 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-042 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-044 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-046 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-047 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-048 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-051 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-052 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-053 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-054 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-055 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-056 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-057 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-065 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-066 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-067 | mathematical_modeling | APITimeoutError | confirmed |
| examples_only | automatic | OR-069 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-079 | mathematical_modeling | ModelOutputTruncated | confirmed |
| examples_only | automatic | OR-098 | code_generation | gurobi_linexpr_add_arguments | confirmed by saved program and TypeError |
| examples_only | automatic | OR-100 | classification_or_mathematical_modeling | misclassification_and_unstated_integrality | classification error and generated integer domain confirmed; intended domain not explicit |
| examples_and_route | automatic | OR-005 | code_generation | unresolved_axis_identifier_mapping | confirmed from saved source data/model/program |
| examples_and_route | automatic | OR-013 | data_transfer | context_length_exceeded | confirmed by API rejection; complete input exceeds the fixed model context |
| examples_and_route | automatic | OR-019 | code_generation | nullable_subset_coefficients | confirmed runtime TypeError; upstream extraction versus lookup cause unresolved |
| examples_and_route | automatic | OR-041 | mathematical_modeling | continuous_domain_assumption | confirmed domain difference; intended-domain convention unresolved |
| examples_and_route | automatic | OR-062 | code_generation | unresolved_customer_axis_mapping | confirmed from saved source data/model/program |
| examples_and_route | automatic | OR-080 | mathematical_modeling | APITimeoutError | confirmed |
| examples_and_route | automatic | OR-083 | mathematical_modeling | worker_task_axes_transposed | confirmed from saved source data/model/program |
| examples_and_route | automatic | OR-086 | code_generation | blending_percentage_parser | confirmed from saved source data/model/program |
| examples_and_route | automatic | OR-098 | code_generation | added_unrequested_wage_objective | confirmed from saved source data/model/program |
| examples_and_route | automatic | OR-100 | classification_or_mathematical_modeling | misclassification_and_unstated_integrality | classification error and generated integer domain confirmed; intended domain not explicit |

Objective mismatches whose semantic origin is unresolved are explicitly marked.
The saved original query, Observation/plan, mathematical model, program and execution
log allow individual review without feeding any reference model or target value to generation.
Some failed mathematical-model requests return before CSVQA evidence is attached to
the case record. Their `csvqa_evidence_unavailable_at_failure` flag is explicit; their
fallback counts are unknown, not claimed zero. Recorded fallback totals are lower bounds
where this occurs. No missing plan or Observation is fabricated or regenerated.

## Retry, fallback and repair records

| method | dataset | repair_count | pipeline_retry_count | http_retry_count | recorded_full_data_fallback_count | csvqa_evidence_unavailable_cases |
|---|---|---|---|---|---|---|
| full | automatic | 0 | 0 | 0 | 19 | 0 |
| full | variants | 0 | 0 | 2 | 5 | 0 |
| full | columns/50pct-S1 | 0 | 0 | 0 | 11 | 0 |
| full | columns/50pct-S2 | 0 | 0 | 0 | 11 | 0 |
| full | columns/50pct-S3 | 0 | 0 | 1 | 11 | 0 |
| full | columns/100pct-S1 | 0 | 0 | 0 | 12 | 0 |
| full | columns/100pct-S2 | 0 | 0 | 0 | 13 | 0 |
| full | columns/100pct-S3 | 0 | 0 | 0 | 11 | 0 |
| full | columns/200pct-S1 | 0 | 0 | 0 | 11 | 0 |
| full | columns/200pct-S2 | 0 | 0 | 1 | 10 | 0 |
| full | columns/200pct-S3 | 0 | 0 | 0 | 9 | 0 |
| rag_only | automatic | 0 | 0 | 52 | 16 | 21 |
| few_shot_only | automatic | 0 | 0 | 1 | 0 | 0 |
| examples_only | automatic | 0 | 0 | 48 | 11 | 26 |
| examples_and_route | automatic | 0 | 0 | 5 | 5 | 2 |

HTTP retries are the original SDK transport policy, not a new model repair or best-of-output run.
When CSVQA evidence is unavailable, its fallback total is a lower bound. Full-model final cases have complete CSVQA evidence.

## Changes and controls

1. Reuse the existing planned CSVQA data path for AP, TP and FLP. Python invokes
   the CSVQA Tool once before one modeling call. This fixes actual skipped-tool
   failures without protocol correction, model reruns or generated-code repairs.
2. Preserve every original field value and row position. Validate matrix axes and
   expose an optional bijection only for declared axes with complete unique matching
   numeric suffix strings. Leading zeros are preserved; unresolved axes fail the plan
   validation and use the existing logged full-source fallback.
3. Reject filters justified only by illustrative examples, separate categorical-code
   prefixes from arbitrary substring matching, and prevent demonstration/sample
   entity counts from narrowing the current data. These rules never inspect a case ID,
   gold objective or reference model.
4. Keep common model-return, Gurobi naming, tuple indexing and logical-constraint
   interface guidance independent of retrieved examples. The query-only Others
   modeling route and its reference source remain; only single-pass parser/client
   controls are shared. No model repair, truncation rerun or solver/code retry exists.
5. Restore the original SDK maximum of two HTTP transport retries and share a
   connection pool after a recorded TLS EOF. Actual HTTP retries are logged through
   `x-stainless-retry-count`, separately from zero pipeline/model retries.
   A common external limit of 1,800 seconds per case prevents indefinitely blocked
   experiments. It was adopted after a saved makespan run exceeded ten million
   branch-and-bound nodes without proof-gap progress. All previously completed final
   cases were below that limit. Solver parameters and matching tolerances stay unchanged;
   interrupted/nonoptimal results count as failures and are never regenerated. The
   adoption record and each method's `runtime_control.json` preserve this decision.
6. RAG Only removes retrieved model/Observation demonstrations for NRM, RA, TP,
   AP and FLP, Others Abstract Model Plan/code demonstrations, retrieved code examples,
   inserted query-only demonstrations and its three fixed Question/Final Answer triggers.
   Query-only ORLM_QA retrieval, classification FileQA evidence and current
   CSVQA loading/profiling, plans, Python extraction/validation and full-data fallback
   remain, as does the Others CSV schema/statistics/runtime reading path.
   Both ablations use the full model's recorded predictions, so classification
   retrieval is retained in definitions but is not rerun during these comparisons.
7. Few-shot Only retains route model/code examples. Python supplies every current
   CSV field directly as text to mathematical modeling. There is no CSVQA Tool,
   planner, LLM extraction/summary, row selection or data rewrite. Code generation
   gets the symbolic variable/objective/constraint sections, query, paths and column names; complete current CSV
   Observations are not forwarded to it. Python removes echoed markdown tables and parameter/Data Mapping
   sections before transfer, without summarizing or rewriting source values. The generated program reads source files.
   This boundary was added after inspecting saved v4 models that echoed complete coefficient tables.
   The v4 Few-shot Only pass (80/101) remains preserved as a preliminary boundary-violating round;
   the complete v5 pass is the final comparison regardless of whether its score improves.
8. Both LOTO notebooks filter the target semantic type from the effective classifier
   reference (`RAG_Examples_All`, which the baseline actually uses), route example
   library and fixed classifier demonstrations. Generic class definitions and model
   interfaces remain. Examples Only retains all routes; Examples And Route reclassifies
   among remaining labels and checks route availability before any formulation.
   The separate query-only reference library is also filtered by its original
   problem-type field, and target-type fixed query-only triggers are removed. The
   full notebook's query-only route remains intact. These query-only exclusions were
   checked offline; the supplied 101-case benchmark has external CSV inputs throughout.

`RUN_AUTOMATIC`, `RUN_VARIANTS`, `RUN_REDUNDANT_COLUMNS`, `RUN_API`, and `RUN_LOTO`
are false in delivered notebooks until explicitly enabled. The 606 switch is only
enabled after the necessary full-model gate and nine-sheet review. Redundant-column
90% is an improvement goal (32/35); each sheet's goal status is in `gate.json`.
The delivered full notebook differs from its evaluated frozen copy only in the
606 explanation/switch cells (38 and 39); all modeling/execution definitions and inputs
are identical. The two delivered LOTO notebooks also correct only the unused query-only
reference filter in cell 39: Knapsack Problem is classified as RA for target-reference exclusion.
All other LOTO definitions match their evaluated copies, and all 101 inputs have CSV files,
so no evaluated execution uses that filter. The corrected no-CSV filter is verified offline
(`query_only_alias_boundary_verification.json`); its API performance is not evaluated.
Experiment provenance and cached classifications refer to the frozen validated copies. `delivery_control.json` records both hashes and the gate snapshot.

## Reproduction and preserved history

Final comparison folders are `full_v4`, `rag_only_v4`, `few_shot_only_v5`, `examples_only_v4` and `examples_and_route_v4`.
`few_shot_v5_adoption.json` preserves the boundary correction and the mandatory complete new-pass policy.
`rag_only_removed_example_candidates.json` lists all 15 effective reference rows (their source indices, types and prompts)
whose model/Observation/code demonstrations are removed; the three query-only fixed triggers are INTEGER, MULTI-PERIOD FLOW and LOGIC+BINARY.
`component_manifest.json` lists retained retrieval functions and separate query-only reference coverage.

Run with `/opt/miniconda3/envs/lean_llm_opt_4_1/bin/python`:

```bash
python scripts/evaluate_minimal_revision_20261005.py --method full --version v4 --workers 10 --run
python scripts/evaluate_minimal_revision_20261005.py --method rag_only --version v4 --workers 3 --run
python scripts/evaluate_minimal_revision_20261005.py --method few_shot_only --version v5 --workers 3 --run
python scripts/evaluate_minimal_revision_20261005.py --method examples_only --version v4 --workers 3 --run
python scripts/evaluate_minimal_revision_20261005.py --method examples_and_route --version v4 --workers 3 --run
python scripts/report_minimal_revision_20261005.py
```

The runner freezes notebook bytes, code/data hashes, paths, settings, model snapshot,
solver threads and matching tolerances before inference. A narrowly checked delivery exception
allows the preserved CSV LOTO pass to be read/resumed after the no-CSV alias-only correction;
changed notebook hashes, replacement fragments and zero query-only cases are recorded.
A fresh experiment version freezes the current corrected notebook. It strips unused gold model
text from worker inputs. Each case has an attempt marker, result, artifacts and log;
every recorded failure is preserved during resume. An interrupted attempt is recorded
as failed and never regenerated. New versions and sheets use distinct directories.
`environment.json` records the interpreter and library versions; original inputs remain
unchanged, with hashes in each manifest. `backups/` contains the five files before editing.

Historical 96/101 and 32/36 results did not match the initial current source fingerprint.
Only the historical additional-benchmark 200pct-S1 records matched it (31/35).
Those historical figures are context, not current-version results. Diagnostic v1/v2/v3
stopped new submissions after evidence of general failures; their attempts and state remain separate,
and none is substituted into v4. A single final pass does not prove repeat-run stability.
