# ReAct review and archived evaluated source: v13

This record supplements the preserved v9/v10 review. It does not replace any earlier
result, source snapshot or gate decision. Current full notebook SHA256:
`164fa23f8d08ffb34e7b32d12bb298eea548d538279acadbf6fca1acc9c2db78`.
v13 has passed offline validation and preflight. All 452 prescribed outcomes are now
recorded: 92/101, 30/36 and 295/315. The three-group mean is 89.3577%, below the
v7 baseline of 93.0866%. The necessary 101 and Variants floors also fail. Three
last-sheet cases hit API credit_balance_exhausted; the coordinator complete flag
remains false, and every failed outcome remains in the denominator. The current
delivered files are the offline-only v14 candidate, whose performance is unverified.

## Confirmed launch policy

The user confirmed the equal mean of three Objective Match accuracies:
`(101_match/101 + variants_match/36 + redundant_match/315)/3`.
The reference is the complete ReAct v7 pass: 94/101, 34/36 and 289/315.
The pooled match count out of 452 is reported separately. Individual dataset, sheet
and class declines are allowed and disclosed. Necessary floors remain 93/101,
92% for each of AP/FLP/NRM/RA/TP, and 32/36 for Variants. All nine redundancy sheets
must be evaluated. Their 31/35 and 32/35 targets remain improvement goals.
The runner blocks study inference until complete overall improvement and these
floors are established. Four isolated gate tests cover permitted individual decline,
the Variants floor, the class floor and incomplete evaluation. Historical gates are
preserved under the policy in effect when they were recorded.

## Evidence for the current minimal changes

v10 Variant30's accepted CSVQA plan ignores 13 files. Each ignored file has the
schema of a selected logical table and contains rows matching the original query's
tenant and date restrictions. This is confirmed from actual source records and the
saved plan, without using a reference model or objective. The selected model receives
an incomplete set of records. `ignored_fragment_reproduction_v10.json` records the
file indices, schemas and matching counts. No problem ID or objective is used in the
new generation or validation logic.

The common plan validator now checks ignored same-schema files against the selected
view's original query filters. Column order does not affect schema equality. An
explicit `table` column separates logical tables when present. If relevant rows
remain, the plan is invalid and the existing exact full-source Observation fallback
is used. The planner receives three lines explaining fragment completeness and
combining logical tables before selecting versions. Source values, row indices and
query semantics are unchanged; no automatic join, revision selection, coefficient
construction or mathematical correction is added.

Six isolated fragment tests reproduce four failures before the correction and all
pass after it. They also verify that other tenants, future dates and different
logical tables can still be ignored, and that fallback retains exact source values
with one planner attempt and zero repair or retry. Existing five shared regressions,
35 protocol/interface checks and 14 LOTO boundary checks pass. All five notebooks
parse and use English prose. All 1,133 input hashes equal the v7 manifest. The five
preserved direct-call notebooks remain byte-identical to their backups.

v10's negative formatting instruction did not prevent degeneration: many completed
101/Variants attempts contain over 32,000 consecutive EM SPACE characters and end
with `finish_reason=length`, after a valid CSVQA Observation. The causal contribution
of that prompt cannot be established from these stochastic runs. v11/v12 replaced
that wording with one positive concise Markdown instruction; v13 removes that extra
sentence after it also failed to prevent degeneration. Sampling, output limits,
transport retries and truncation scoring remain unchanged. A performance benefit
from the formatting changes has not been established.

## Last shared parsing instruction before inference

Inspection of the actual v10 source text and generated parser confirmed identifier
substring collisions: a short identifier matches the suffix of a longer one, giving
the same grade both a 50% lower bound and a 10% upper bound while positive production
is required. `blending_parser_inspection_v10.json` records the exact parser results.
An initial speculative operand-order explanation was rejected after reading the source.
The shared code prompt now requires complete identifier matching and escaping regex
identifiers. No dedicated parser, benchmark value or problem-specific model patch was
added. This prompt improvement is not guaranteed and awaits the complete pass.

v11 is preserved as a preflight-only candidate with zero inference cases. v12 adds
only the shared three-line parsing instruction to it. Five interface regressions,
35 ReAct checks, six fragment regressions and 14 LOTO boundaries pass for v12.
The user-policy gate tests also pass. All input hashes equal v7. The original direct
notebooks remain unchanged. Notebook version/configuration identifiers and study
base-source hashes are updated consistently. There is one current general full source
for every dataset; no separate dataset-specific generation code exists.

## Lossless scalar-filter adaptation after the v12 pass

The v12 accepted extraction schema allows scalar or list values, but prefix, equality
and scalar numeric/date operators could not accept a one-element list. A saved plan
requests the exact query prefix as a one-element list and causes `str.startswith`
to raise TypeError. The existing fallback then supplies every raw row; the generated
model loses the query subset. This is a data-interface defect, not an incorrect
coefficient or a reference-answer discrepancy.

v13 adds four lines: unwrap a singleton for scalar operators, reject ambiguous
multi-value scalar forms, and preserve list semantics for in/not_in/between. Two
planner lines explain the contract. Five isolated tests reproduce four errors before
and all pass after the correction. A source-only check of the real saved plan now
returns the one matching original row out of seven, with the original source row and
values unchanged. No model/code generation or objective rescoring occurs in this check.
`singleton_filter_source_reproduction_v13.json` preserves the evidence.

The additional positive formatting sentence also failed to prevent output degeneration
in v12 and is removed. The original ReAct demonstration/model format remains, together
with the general unit, conservation and activation guidance. No model parameter,
truncation scoring, repair or outcome retry is changed. v12's complete pass is retained;
v13 is the next common source for all datasets and studies. Its independent pass subsequently records 92/101, 30/36 and 295/315; the
necessary floors and overall improvement gate fail. All outcomes and errors are retained.

## Shared route implementation

Every CSV route uses the original `ZERO_SHOT_REACT_DESCRIPTION` agent. The agent
must execute CSVQA for current data before its first accepted Final Answer. The
existing planner returns a declarative extraction plan; Python applies allowlisted
query-supported filters and preserves exact source strings. The tool's Observation
contains table IDs, file indices, columns, source rows and any verified matrix-axis
mapping. Validated records also become indexed pandas `CSVQA_FRAMES` for execution.
The final mathematical model is symbolic and supplies an exact Data Mapping.
Full/RAG/LOTO code generation receives the query, symbolic model, exact structured
CSVQA Observation and interface guidance. The generated program uses source-backed
frames supplied by Python, and extraction accepts only a live optimal Gurobi model
with a finite objective. Numeric CSVQA records can be present in these code prompts;
Few-shot Only instead receives source paths/headers and excludes the complete
numeric Observation. A local captured-prompt check verifies that distinction. Static source normalization protects non-ASCII display names
unless the generated program uses named variable/constraint lookup.

| Route | Modeling examples | Additional guidance |
| --- | --- | --- |
| AP | nearest 1 AP example | query-defined assignment domains and complete source axes |
| FLP | nearest 1 FLP example | facility/customer axes, costs, capacities and opening links |
| NRM | nearest 1 NRM example | symbolic sets, conservation constraints and exact source mapping |
| RA | nearest 3 RA examples | every required resource dimension, business IDs, domains and resource/activity indexing |
| TP | nearest 3 TP examples | source order, complete symbolic transportation model and exact mapping |
| Others / Mixture | nearest 3 general/mixed examples | interacting decision families, scheduling boundaries, units, summation scopes and source-defined axes |

Example counts are retrieval limits, not current instance sizes. The original query
controls every formulation. The common unit/balance/activation and Gurobi API guidance
added in v10 remains. No route is selected from the reference objective. The original
Others-without-CSV formulation function is unchanged; its live performance has not
been tested by these entirely CSV-based benchmarks. Shared code-generation guidance
also applies to query-only programs; their self-contained data interface passes an
offline test.

## Research variants derived from the same full code

RAG Only reuses the final full classification cache and removes the complete 15-row
modeling/Observation/code example library plus fixed query-only demonstrations.
It retains CSVQA, profile retrieval, the planner, Python filtering, completeness and
matrix validation, full-source fallback and the query-only ORLM_QA retrieval path.
The detailed demonstration-removal inventory remains preserved with this project.

Few-shot Only also reuses the final full classification cache. It retains route
modeling and code examples, removes CSVQA and its planning/extraction path, and loads
all current source CSV records directly with Python. The complete unchanged source
Observation is supplied to the ReAct modeler without tools. Code generation receives
only the symbolic model and source schemas/paths; execution reads those source files.
The complete numeric Observation is never sent to code generation.

LOTO Examples Only removes the held-out type from classification references and all
modeling/code reference retrieval, then classifies again while retaining every route.
The type's original route can model with generic instructions and current data when
its reference pool is empty. Shared model-return and execution constraints remain.

LOTO Examples And Route additionally removes the target workflow from the classifier's
allowed choices. The classifier chooses among the remaining workflows using only the
question and remaining references. A guard rejects any disabled route before modeling.
The true type defines the held-out fold and scoring only; it never selects a substitute
route. Mixture and Others share the Others workflow, so either held-out fold disables
Others in this setting. No excluded example or reference answer is reintroduced.

These definitions pass offline boundary checks. The completed v10 and v12 passes
did not satisfy the necessary floors and overall comparison. No new ablation or
LOTO inference has been launched. The v13 pass has recorded all 452
outcomes but fails the 101 and Variants floors and the overall comparison.
Studies remain blocked. The subsequent v14 candidate only removes one conflicting
common prompt statement and has zero inference cases because API credit is exhausted.
The earlier direct-call study results are separate historical results.

## Execution and reproducibility

Each code version is frozen and every case has one scored pipeline attempt. A malformed
ReAct envelope, missing mandatory CSVQA or malformed Action Input may restart the agent
under the user's explicit policy, within the same 1,800-second deadline. Every discarded
protocol attempt is recorded. The first protocol-valid completion is retained. Truncated
responses, generated-code errors, solver failures and objective mismatches never trigger
another attempt and remain in all denominators. Transport retries retain the existing
SDK policy and are logged separately. Full-source fallback is a data path, not model or
code repair. There is no repair agent or outcome-driven retry.

Fixed configuration: gpt-4.1-2025-04-14, text-embedding-ada-002, temperature 0,
top_p 1, Gurobi Threads 2, MIPGap 1e-4, rel_tol=abs_tol=1e-4. Backups, frozen
notebooks, input hashes, per-case logs, CSVQA plans/Observations, symbolic models,
generated code, solver results and failure annotations are retained by version.
606 is disabled and unexecuted in this work. A single complete pass cannot establish
cross-round stability, and future unseen-dataset performance is not guaranteed.
