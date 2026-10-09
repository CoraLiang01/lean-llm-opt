# Revised Few-shot Only: one complete 101-case evaluation

All cases were computed once under the revised source. Classification predictions were reused from full ReAct v7. Failures, truncation, context errors and 1,800-second deadlines remain in the denominator. No best-of-round selection is applied.

| Experiment | Classification | Optimal solve | Objective Match | OM percent |
|---|---:|---:|---:|---:|
| Full ReAct v7 (existing) | 94/101 | 96/101 | 94/101 | 93.07% |
| Few-shot Only v7 (previous) | 94/101 | 75/101 | 63/101 | 62.38% |
| Few-shot Only direct model (new) | 94/101 | 91/101 | 86/101 | 85.15% |

| Class | Full v7 OM | Previous Few-shot OM | Revised classification | Revised solve | Revised OM |
|---|---:|---:|---:|---:|---:|---:|
| AP | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| FLP | 13/14 | 5/14 | 14/14 | 12/14 | 12/14 |
| NRM | 25/25 | 18/25 | 25/25 | 23/25 | 21/25 |
| RA | 22/22 | 12/22 | 22/22 | 22/22 | 22/22 |
| TP | 9/9 | 8/9 | 9/9 | 8/9 | 8/9 |
| Others | 5/8 | 4/8 | 7/8 | 7/8 | 6/8 |
| Mixture | 15/18 | 11/18 | 12/18 | 14/18 | 12/18 |

Execution audit:

```json
{
  "cases": 101,
  "new_computed_cases": 101,
  "cached_classification_cases": 101,
  "csvqa_calls": 0,
  "repair_count": 0,
  "protocol_restart_count": 0,
  "pipeline_retry_count": 0,
  "http_retry_count": 5,
  "fallback_count": 0,
  "missing_observation_trace_cases": 1,
  "input_files_checked": 298,
  "input_changes": [],
  "baseline_input_files_checked": 1133,
  "baseline_input_changes": [],
  "historical_sources_results_checked": 206,
  "historical_changes": [],
  "source_sha256": "5ddad9b79a56ba8572d74f743afee427e23d6237567f19eb1ea2dfbd1fc32a81",
  "complete": true
}
```

The revised notebook forwards the formulation unchanged, replaces current-case CSVQA with complete Python Observation for modeling, and does not separately forward Observation to code generation. Shared ReAct v7 prompts/examples, model configuration, solver settings, matching tolerance and no-CSV route remain unchanged apart from necessary data-source instructions.

The source/interface changes were evaluated together. Differences from the earlier pass include model-output variation; they cannot all be causally assigned to one change. A single complete pass does not establish cross-run stability.

Failure-stage causes are marked confirmed only when supported by saved query/model/code/input evidence. Otherwise they remain unconfirmed. Each raw model, code, log and result is retained in automatic/attempts/<problem_id>.

Failures:

- OR-005 (TP): modeling_data_mapping_and_code_generation; source_identifier_alias_not_mapped.
- OR-013 (NRM): formulation_input; context_length_exceeded.
- OR-015 (NRM): code_generation; exact_identifier_match_used_for_family_code.
- OR-023 (NRM): code_generation; fractional_revenue_truncated_to_integer.
- OR-024 (NRM): code_generation; fractional_revenue_truncated_to_integer.
- OR-062 (FLP): data_mapping_and_code_generation; unresolved_store_identifier_mapping.
- OR-073 (Mixture): code_generation; validation_of_query_forbidden_equipment_product_pairs.
- OR-080 (Mixture): mathematical_model_and_code_generation; incorrect_time_boundary_and_minimum_down_time.
- OR-084 (Others): solve; external_deadline_without_optimality_proof.
- OR-086 (Mixture): code_generation; source_text_parser_requires_unstated_delimiters.
- OR-087 (Mixture): mathematical_model_unconfirmed; objective_mismatch_shared_capacity_semantics.
- OR-090 (Mixture): code_generation; pandas_series_index_realignment_creates_nan.
- OR-097 (Mixture): code_generation; gurobi_none_variable_bounds.
- OR-099 (FLP): data_mapping_and_code_generation; warehouse_identifier_alias_not_mapped.
- OR-100 (Others): classification_and_mathematical_model_unconfirmed; objective_mismatch_integer_domain_interpretation.

The one missing direct-source Observation trace is OR-013, rejected for excessive context length before the modeler returned. CSVQA remains disabled by code; this is not an unknown CSVQA attempt. HTTP retries were recorded for OR-042, OR-043, OR-045 (one each) and OR-101 (two); all four ultimately matched. Retry reasons/status codes were not captured and are not inferred.
