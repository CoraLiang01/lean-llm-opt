# Current ReAct v7 review

This is a supplementary read-only review. No notebook or raw evaluation record was
modified, and no API case was run during this review. Earlier reports remain intact.

## Identity and measured scope

Current full notebook: LEAN_LLM_OPT_4.1_Large-scale_1006_ReAct.ipynb.
Evaluated frozen source: full_v7/frozen_notebook.ipynb.
The later filename separation changed only self/base notebook filename assignments;
modeling and execution behavior remains the evaluated v7 behavior.

The complete prespecified v7 pass contains 452 cases: 101 automatic, 36 variants,
and all 9 redundant-column sheets with 35 cases each. Failures remain in denominators.

| Dataset | Classification correct | Optimal returned model | Objective Match |
| --- | --- | --- | --- |
| 101 | 94/101 | 96/101 | 94/101 |
| Variants | 32/36 | 34/36 | 34/36 |
| 50pct-S1 | 33/35 | 33/35 | 33/35 |
| 50pct-S2 | 33/35 | 33/35 | 33/35 |
| 50pct-S3 | 33/35 | 33/35 | 33/35 |
| 100pct-S1 | 33/35 | 32/35 | 32/35 |
| 100pct-S2 | 33/35 | 32/35 | 30/35 |
| 100pct-S3 | 33/35 | 33/35 | 32/35 |
| 200pct-S1 | 33/35 | 32/35 | 32/35 |
| 200pct-S2 | 33/35 | 33/35 | 33/35 |
| 200pct-S3 | 33/35 | 33/35 | 31/35 |

| 101 class | Classification correct | Optimal returned model | Objective Match |
| --- | --- | --- | --- |
| AP | 5/5 | 5/5 | 5/5 |
| FLP | 14/14 | 13/14 | 13/14 |
| NRM | 25/25 | 25/25 | 25/25 |
| RA | 22/22 | 22/22 | 22/22 |
| TP | 9/9 | 9/9 | 9/9 |
| Others | 7/8 | 6/8 | 5/8 |
| Mixture | 12/18 | 16/18 | 15/18 |

Compared with the preserved direct-call family, 101 Objective Match falls from
97/101 to 94/101 (-2.97 percentage points). Its semantic predictions are identical
case by case. Variant Objective Match remains 34/36. Redundancy falls from 303/315
to 289/315 (-4.44 percentage points). These are paired observed outcomes, not a
controlled estimate of the causal effect of ReAct alone.

101 improvement: OR-080. Regressions: OR-050, OR-062, OR-074, OR-097.
Variant improvements: Variant29 and Variant36; regressions: Variant1 and Variant16.

Necessary gate is passed, but sheet improvement goals are not all met:
100pct-S2 is below 31/35 and 90%; 200pct-S3 is below 90% and the preferred 32/35.
FLP clears the 92% minimum but misses the 95% aspiration. One pass does not
establish repeated-run stability. 606 remains disabled and unexecuted.
The four ReAct ablation/LOTO campaigns are not API-evaluated; offline checks are
not their performance results. The older 77/81/73/91 study scores belong only to
the preserved direct-call family.

## Actual architecture and boundaries

Classification uses ZERO_SHOT_REACT_DESCRIPTION and FileQA over five similar
prompt/type references. Mixture and Others map to the Others workflow.
NRM, RA, TP, AP and FLP modeling uses the original ReAct agent class with CSVQA.
The agent itself invokes the tool; Python does not pre-call CSVQA before the agent.
The accepted current invocation must contain at least one CSVQA call. The current
Observation state, original file-index mapping and intermediate steps are retained.
CSVQA_MODE=planned describes the declarative extraction tool, not replacement of
ReAct modeling by a direct LLM call.

CSVQA plans source columns, explicit filters and relationships from profiles and
the original query. Python extracts exact records and validates declared matrix
axes. Invalid plans use the existing logged complete-source fallback, without a
second planner or solver repair. Multiple calls preserve original zero-based file
indices and combine views by table_id; the latest same-ID view wins.

The full code generator receives the symbolic model/Data Mapping, original query,
structured current CSVQA payload, route code references and shared solver contract.
Python supplies CSVQA_FRAMES from source values and places source_row in the index.
CSVQA_DATA remains available for metadata. The full payload is still included in
the full-model code-generation prompt; the no-full-data transfer restriction applies
specifically to Few-shot Only.

Others with CSV remains the original schema-preview -> Abstract Model Plan LLMChain
-> code LLMChain workflow. Runtime code reads complete source CSV files. This is not
ReAct modeling and does not invoke CSVQA. Consequently, an all-CSV-routes mandatory
CSVQA requirement is not satisfied by Others with CSV in the current implementation.
Others without CSV retains its ORLM_QA ReAct workflow and query-only code generator.
It has no CSVQA requirement. The all-CSV campaign does not API-validate that branch.

The effective example corpus contains 15 rows: AP=1, UFLP/FLP=1, NRM=1, RA=1,
TP=1, Mixture=8 and Others=2. Canonical modeling requests k=1 for AP/FLP/NRM and
k=3 for RA/TP, capped at available same-route rows. Code references request k=2,
also capped. Others modeling requests three from its normalized ten-row pool.

## Changes relative to the preserved direct-call family

1. Restore actual canonical CSV ReAct modeling instead of a pre-called CSVQA plus
   direct model-generation call. Keep the existing route functions and examples.
2. Add the user-authorized protocol wrapper and current-data state. Missing CSVQA,
   malformed agent/Action Input output and incomplete Final Answer can restart only
   the agent within the case deadline. Keep the first protocol-valid completion.
3. Support selected current file indices and preserve their original identities;
   combine successive Observations and retain all tool-call traces.
4. Add the source-only DataFrame boundary in v7 and its shared generation instructions.
5. Add generic Gurobi instructions for numeric/omitted bounds, distinct decision
   container names, iterable quicksum inputs, source-string CSV reading, proper
   constraints and unconditional original-query bounds. Others retains its two-chain
   flow and receives the additional unconditional-bound modeling instruction.
6. Keep scoring, objective tolerances, model-return validation and solver settings.
   Preserve both notebook families and separate their versioned outputs.

Extraction planning, exact source authority, matrix validation, much of the solver
contract and the existing source-code normalization were inherited from the
preserved direct-call family; they are not all new ReAct changes.

## Supplementary confirmed failure evidence

The earlier summary categorizes OR-074 and Variant16 as absent variable-name lookups.
This review identifies the actual conflicting components: raw LLM output creates
bulk variables with name='x' and later uses getVarByName('x[...]'). The inherited
_source_candidate AST normalization changes addVars/addConstrs names to ''. It is
identical in the preserved direct-call and current ReAct notebooks. The saved
execution program therefore has anonymous variables but still searches for x[...],
producing None and then AttributeError. This is a generic postprocessing/generated
program interface failure, not evidence of an incorrect objective formulation.

OR-097 is more specific than its earlier execution/TypeError label: the raw program
and saved execution source pass lb=None and ub=None to addVars. This is a confirmed
code-generation API error. It is not an unresolved upstream semantic explanation.

Other automatic failures: OR-050 reads source_row as a DataFrame column even though
it is the source index; OR-062 lacks a justified customer/city matrix-axis mapping;
OR-084 exceeds the unchanged 1,800-second case deadline; OR-087 and OR-100 are
objective mismatches whose complete semantic explanations remain unresolved.
Variant1 is truncated while producing the Others Abstract Model Plan.

No new generation patch is applied in this review. These cases remain failed in
all reported metrics. No completed objective is salvaged from a failed program.

Two unresolved prompt inconsistencies remain: NRM/RA still say CSVQA exactly once
inside their route prefix while the shared policy permits multiple calls, and TP
still requests a numerical formulation while the shared suffix requests a symbolic
model. They are observable source inconsistencies, not confirmed causes of a
particular scored failure.

The ReAct protocol policy covers the five canonical CSV modeling agents. It does
not force the legacy Others CSV two-chain workflow to call CSVQA. Few-shot Only
removes CSVQA by experimental design.

## Experiment preparation

RAG Only removes inserted formulation/Observation/code demonstrations from the
15-row library and the fixed query-only demonstrations, but retains CSVQA profiling,
planning, deterministic extraction/validation/fallback and query-only ORLM_QA.
Ablation classification reuses the frozen ReAct full-model classification cache.

Few-shot Only retains reference examples. Python reads every original CSV field
and row as a complete modeling Observation. CSVQA and its extraction planner are
absent. A deterministic transfer filter removes full parameter/Data Mapping sections
and numerical tables before code generation; that stage receives the symbolic
model, query and source information and reads source files at runtime. Its canonical
modeling agents retain the ReAct framework with an empty data-tool list.

LOTO Examples Only removes the held-out semantic type from classification, modeling
and code reference material, including relevant fixed/query-only examples, then
reclassifies. Its workflow route remains available.
LOTO Examples And Route additionally removes the workflow from allowed classifier
labels and validates the route before modeling. Disabling Others excludes both
Others and Mixture labels. True labels define folds/exclusions and scoring, never
replacement routing. Test reference models/objectives are not generation inputs.

## Retry and reproduction record

Measured v7 logs: model/code repair=0, full-pipeline retries=0, protocol restarts=0,
HTTP retries=0, recorded complete-source data fallbacks=117. The authorized restart
logic was tested offline but not triggered in the measured v7 pass. Inherited fixed
AST source normalization still modifies code before execution; repair_count=0 must
not be interpreted as no source postprocessing.

Model gpt-4.1-2025-04-14; embeddings text-embedding-ada-002; temperature=0; top_p=1;
Gurobi Threads=2; MIPGap=1e-4; common case deadline=1,800 seconds;
Objective Match uses rel_tol=abs_tol=1e-4. SDK max_retries=2 is configured.
The manifest/environment records and all 1,133 input-file hashes are retained.

Read report_v7/all_case_results.csv, report_v7/failures.csv, full_v7/manifest.json,
full_v7/environment_record.json and the per-case generated model/source/log artifacts.

Current renamed notebook SHA256: 7bb36bb65963ecff267b1e1ee607b9128fe26390465290770f8cb255438ad266
