# Evidence-based ReAct review and minimal revision

The reference baseline is the complete frozen v7 pass (94/101, 34/36, 289/315).
The preserved direct-call family remains unchanged and its 97/101 is a separate control.
All five ReAct notebooks, all definition cells, experimental overrides, scoring,
cache/provenance code and the evaluation runner were reviewed. Per-notebook cell
diffs and SHA256 identities are retained beside this report.

## Confirmed regressions and contradictions

- OR-074 and Variant16: generated named-variable lookup is incompatible with
  unconditional anonymous-name AST normalization. Retain the original normalization
  only when the program does not query variable/constraint names. Business keys,
  coefficients and optimization expressions are unchanged.
- Diagnostic v8 removed all normalization and exposed non-ASCII solver-name decoding
  failures (OR-012, OR-021). A synthetic cafe-key Gurobi reproduction failed before
  the correction and passed afterwards. v8 was stopped after 47 submitted cases;
  its records are preserved and never reused as v9 computations.
- OR-050 and redundant-column errors: source_row is the DataFrame index; frames do
  not expose nested record dictionaries or .records. Add one concrete shared runtime
  interface description; retain exact original source records and frame construction.
- NRM/RA exactly-once wording conflicted with permitted multiple CSVQA calls; TP
  numerical-model wording conflicted with the symbolic-model boundary. Remove those
  stale route-level requirements, leaving the common ReAct contract authoritative.
- Others CSV bypassed CSVQA (102/452 v7 cases). Reuse the existing shared ReAct
  formulation/code path instead of its two LLMChains. Retain its reference pool and
  generic objective, filtering, scheduling, summation and matrix-boundary guidance.
  The separate query-only ORLM_QA formulation function is unchanged.
- Invalid file_indices were a ValueError outside the allowed protocol restart path.
  Classify malformed Action Input as a protocol error; filesystem/API errors do not
  become protocol retries. Keep the original source file indices and complete traces.
- Few-shot Only's external-CSV generation instruction also applied to query-only
  get_code. Select the explicit direct-source instruction only from get_csv_code;
  query-only code retains the self-contained data contract. No full CSV Observation
  is forwarded to Few-shot Only code generation.
- LOTO's separate query-only reference filter mapped standalone Knapsack Problem to
  RA despite the classifier's Others definition. Exclude it with Others, retaining
  explicitly labeled Resource Allocation Problem as RA. No query-only inference
  route is changed by this reference-boundary correction.
- Report code counted protocol restarts again as whole-pipeline retries. Subtract
  protocol counts from that reporting column; keep both counts and HTTP retries.
- Add a complete-pass comparison gate to the experiment launcher. API ablations/LOTO
  cannot start until 101, Variants, all nine sheets and every 101 class do not decline
  in Objective Match against v7. All inputs and denominators must remain identical.

## Retained uncertainty and controls

OR-062 and several OR-023 sheet instances lack a justified customer/city axis mapping.
No mapping is fabricated from objectives or answers. OR-087/OR-100 semantic mismatches
remain unresolved. Raw truncated responses are now retained in run.log for diagnosis,
but never accepted, regenerated or used for outcome selection. The diagnostic v8
OR-009 truncation occurred in formulation after a valid CSVQA Observation; a complete
semantic cause cannot be inferred from an absent response.

The route-name normalizer, classifier definitions/reference prompts, source filters,
matrix validation, scoring tolerances, Gurobi model-return resolution, solver settings
and immutable attempt/resume mechanism remain. No repair agent, generated-code repair
following an error, solver retry or objective-based retry is introduced. The inherited
static solver-name normalization is explicitly recorded as pre-execution normalization.

## Validation before the full v9 pass

Five local regressions pass, including real tiny Gurobi solves and actual ReAct/CSVQA
integration with a local fake chat service. The existing 35 protocol/source-transfer
checks pass. Fourteen LOTO fold/exclusion/availability checks pass. All five notebooks
parse and all source prose is English. API calls in these checks: zero. These checks
validate interfaces and boundaries; they do not establish benchmark performance.

The full v9 pass is independent: one scored pipeline attempt for each of 452 cases,
with the user's allowed protocol restarts inside a case. Failures and truncations
stay in all denominators. Fixed snapshot gpt-4.1-2025-04-14, temperature=0, top_p=1,
Gurobi Threads=2, MIPGap=1e-4, rel_tol=abs_tol=1e-4 and a common 1800-second deadline.
606 remains disabled and unexecuted. New research studies start only after the
prespecified comparison gate passes.

## Source-semantic caveat found during v9 inspection

OR-062 now numerically matches, but generated code aligns demand rows and city columns
by position. An explicit Customer-to-city mapping is still not established from the
question/source. This match is retained under the unchanged numeric scorer and is
separately marked as an unverified modeling assumption, not a proven mapping fix.

## Confirmed generated-response failures during the frozen v9 pass

OR-086, OR-087 and OR-095 produce over 32,000 consecutive Unicode EM SPACE
characters during formulation after a valid CSVQA Observation. Reference prompt/model
fields contain no repeated Unicode-space runs. This is response degeneration and
output-limit exhaustion, not missing CSVQA data. Truncated responses remain failures.
Variant30 uses gp.min_ in an inequality; Gurobi general expressions require equality
to an auxiliary variable. This is a confirmed generated API error, not a solver
optimality issue. No change or rerun is applied inside the frozen v9 pass.

## Subsequent v10 corrections, pending performance validation

The frozen v9 pass remains unchanged and continues through all nine sheets. v10 adds
only shared prompt guidance in formulation and code generation: ordinary ASCII
formatting (without changing source identifiers), source-derived interval units,
consistent conservation balances, unconditional bounds, required activation links,
legal Gurobi general expressions/binary indicator triggers and omitted infinite
upper bounds. Model/solver parameters, data interfaces and scoring are unchanged.

Variant10's generated root balance has the wrong sign and nonzero total balance.
Variant32 conditions explicitly unconditional category limits on activation.
Variant13 passes ub=None despite the existing interface guidance.
Variant34 uses an integer quantity as an indicator trigger; a synthetic Gurobi test
returns quantity 7 for a minimization whose valid linked-binary model returns 2.
The saved v7 generated solution satisfies every v9 linear constraint and bound; the
added nonbinary indicator constraints are the observed structural incompatibility.
No benchmark optimize call or rescoring was used in this read-only reconstruction.

The earlier suggestion of direct binary coercion was provisional and corrected:
local exposed VType remains integer, while solution behavior changes. Only the
confirmed invalid trigger/interface issue is used to guide the prompt correction.
Gurobi documents binary indicator triggers at:
https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.addGenConstrIndicator

The shared instruction about unconditional bounds also needs logical activation
links to remain explicit: 50pct-S2/OR-025 incorrectly omitted all supplier activation
links when no physical supplier capacity was given. The revised prompt distinguishes
logical operating links from unconditional physical quantity bounds.
50pct-S1/OR-033 interpreted 8 working hours as 8 half-hour rows. Its revised guidance
requires interval durations from time labels without adding any case-specific values.

Variant32 also selects latest revisions independently per file before concatenating
fragments. v10 adds one general instruction to combine the logical table first and
apply the query-defined version/deletion/deduplication order before aggregation.
This is supported by the saved generated code, not by reference model inspection.

The final v10 source SHA256 is 9850407baf1d558de648e3f07489ca1b5650e4698857878a24067e43f640b5fc.
Its five regressions, 35 protocol tests and 14 LOTO fold/guard checks pass locally.
All 1,133 input hashes equal the v7 manifest. The preserved five direct-call notebooks
are byte-identical to their backups. Only external-CSV cases occur in these benchmarks;
query-only formulation source is unchanged, but its live API performance remains
unverified. Common code-generation API guidance also applies to query-only programs.

The first v10 response-degeneration failure is OR-005: valid CSVQA is followed by
32,601 consecutive EM SPACE characters despite the new formatting guidance. It remains
a scored truncation failure; no retry or reclassification as a protocol error occurs.
This proves the prompt-only formatting improvement is not a guaranteed prevention.
