# ReAct architecture evaluation: v9

Status: full-model gate not yet passed.
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

## Full 101 by class

| class | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| AP | 5 | 5 | 5 | 5 | 5 |
| FLP | 14 | 14 | 14 | 14 | 14 |
| NRM | 25 | 25 | 25 | 25 | 25 |
| RA | 22 | 22 | 22 | 22 | 22 |
| TP | 9 | 9 | 9 | 9 | 9 |
| Others | 8 | 8 | 7 | 8 | 7 |
| Mixture | 18 | 18 | 12 | 14 | 14 |

## Full datasets and every sheet

| dataset | expected | recorded | classification_correct | solved | objective_match |
|---|---|---|---|---|---|
| automatic | 101 | 101 | 94 | 97 | 96 |
| variants | 36 | 36 | 31 | 29 | 28 |
| columns/50pct-S1 | 35 | 35 | 33 | 35 | 34 |
| columns/50pct-S2 | 35 | 35 | 33 | 32 | 31 |
| columns/50pct-S3 | 35 | 35 | 33 | 34 | 34 |
| columns/100pct-S1 | 35 | 35 | 33 | 34 | 34 |
| columns/100pct-S2 | 35 | 35 | 33 | 33 | 33 |
| columns/100pct-S3 | 35 | 35 | 33 | 35 | 32 |
| columns/200pct-S1 | 35 | 35 | 33 | 35 | 33 |
| columns/200pct-S2 | 35 | 35 | 33 | 34 | 33 |
| columns/200pct-S3 | 35 | 35 | 33 | 34 | 33 |

## Ablations and LOTO against this ReAct baseline

| method | version | recorded | classification_correct | solved | objective_match | delta_matches_vs_full | delta_percentage_points | status |
|---|---|---|---|---|---|---|---|---|
| full | v9 | 101 | 94 | 97 | 96 | 0 | 0.0 | complete |
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
| full | code_generation | 3 |
| full | execution | 6 |
| full | mathematical_modeling | 14 |
| full | mathematical_modeling_and_code_generation | 1 |
| full | mathematical_modeling_or_code_generation | 7 |

| method | dataset | problem_id | failure_stage | failure_type | cause_certainty |
|---|---|---|---|---|---|
| full | automatic | OR-086 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | automatic | OR-087 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | automatic | OR-095 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | automatic | OR-097 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | automatic | OR-100 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/100pct-S1 | 100pct-S1/OR-023 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| full | columns/100pct-S2 | 100pct-S2/OR-021 | mathematical_modeling | ModelOutputTruncated | confirmed |
| full | columns/100pct-S2 | 100pct-S2/OR-023 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| full | columns/100pct-S3 | 100pct-S3/OR-023 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/100pct-S3 | 100pct-S3/OR-025 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/100pct-S3 | 100pct-S3/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/200pct-S1 | 200pct-S1/OR-009 | mathematical_modeling | continuous_instead_of_count_domain | Confirmed continuous domain from saved formulation; query describes counts and also scale, so intended semantics are not independently settled by numeric mismatch. |
| full | columns/200pct-S1 | 200pct-S1/OR-023 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/200pct-S2 | 200pct-S2/OR-023 | execution | KeyError | confirmed; upstream semantic cause may be unresolved |
| full | columns/200pct-S2 | 200pct-S2/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/200pct-S3 | 200pct-S3/OR-023 | execution | KeyError | confirmed; upstream semantic cause may be unresolved |
| full | columns/200pct-S3 | 200pct-S3/OR-028 | mathematical_modeling_or_code_generation | objective_mismatch | unresolved; solver optimality does not prove formulation correctness |
| full | columns/50pct-S1 | 50pct-S1/OR-033 | mathematical_modeling | shift_duration_interval_unit_mismatch | confirmed from saved response/code; 8 hours incorrectly represented by 8 half-hour intervals |
| full | columns/50pct-S2 | 50pct-S2/OR-020 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | columns/50pct-S2 | 50pct-S2/OR-023 | execution | RuntimeError | confirmed; upstream semantic cause may be unresolved |
| full | columns/50pct-S2 | 50pct-S2/OR-025 | mathematical_modeling | supplier_activation_links_omitted | confirmed from saved response/code; all shipment-to-opening links omitted because physical supplier capacity is unspecified |
| full | columns/50pct-S2 | 50pct-S2/OR-032 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | columns/50pct-S3 | 50pct-S3/OR-023 | execution | ValueError | confirmed; upstream semantic cause may be unresolved |
| full | variants | Variant1 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | variants | Variant3 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | variants | Variant10 | mathematical_modeling | inconsistent_flow_conservation | confirmed from saved response/code; root RHS has wrong sign; total closed-network balance is nonzero |
| full | variants | Variant13 | code_generation | none_variable_upper_bound | confirmed from saved response/code; addVars receives ub=None |
| full | variants | Variant21 | mathematical_modeling | unicode_space_output_degeneration | confirmed from saved response/code; valid CSVQA preceded >32000 padding characters and finish_reason=length |
| full | variants | Variant30 | code_generation | invalid_gurobi_general_expression | confirmed from saved response/code; gp.min_ appears in an inequality rather than an auxiliary equality |
| full | variants | Variant32 | mathematical_modeling_and_code_generation | conditional_bound_and_fragment_version_selection | confirmed from saved response/code; unconditional category bounds multiplied by activation; latest revisions selected per file before concatenating fragments; relative contributions unresolved |
| full | variants | Variant34 | code_generation | nonbinary_indicator_trigger | confirmed from saved response/code; integer quantity used as indicator; saved linear constraints admit the v7 generated solution; tiny independent test changes objective from 2 to 7 |


Unresolved objective mismatches remain unresolved until the saved model/program/source
records support an explanation. Missing CSVQA evidence is explicitly marked, and
fallback totals are lower bounds when evidence is unavailable. No trace is fabricated.

## Repairs, retries and fallbacks

| method | dataset | repair_count | pipeline_retry_count | protocol_retry_count | http_retry_count | recorded_full_data_fallback_count | csvqa_evidence_unavailable_cases |
|---|---|---|---|---|---|---|---|
| full | automatic | 0 | 0 | 0 | 0 | 23 | 0 |
| full | variants | 0 | 0 | 0 | 1 | 7 | 0 |
| full | columns/50pct-S1 | 0 | 0 | 0 | 0 | 12 | 0 |
| full | columns/50pct-S2 | 0 | 0 | 0 | 0 | 12 | 0 |
| full | columns/50pct-S3 | 0 | 0 | 0 | 0 | 11 | 0 |
| full | columns/100pct-S1 | 0 | 0 | 0 | 0 | 12 | 0 |
| full | columns/100pct-S2 | 0 | 0 | 0 | 0 | 13 | 0 |
| full | columns/100pct-S3 | 0 | 0 | 0 | 0 | 10 | 0 |
| full | columns/200pct-S1 | 0 | 0 | 0 | 0 | 12 | 0 |
| full | columns/200pct-S2 | 0 | 0 | 0 | 0 | 11 | 0 |
| full | columns/200pct-S3 | 0 | 0 | 0 | 0 | 9 | 0 |


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
python scripts/evaluate_react_revision_20261006.py --method full --version v9 --workers 6 --run
python scripts/evaluate_react_revision_20261006.py --method rag_only --version v9 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method few_shot_only --version v9 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_only --version v9 --workers 3 --run
python scripts/evaluate_react_revision_20261006.py --method examples_and_route --version v9 --workers 3 --run
python scripts/report_react_revision_20261006.py --version v9
```

Every case/version has one scored pipeline attempt. This version's protocol restarts
are logged within that attempt. Resuming preserves failures as well as successes.
Changed code or data requires a fresh result directory. Every manifest freezes the
notebook, input hashes, fixed model snapshot and execution settings. Source files,
model/code artifacts and logs are retained. 606 stays disabled until this revision
passes the required full-model gate and all nine sheets have been reviewed.
