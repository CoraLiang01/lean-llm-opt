# Current ReAct candidate: v14 (performance unverified)

The five delivered ReAct notebooks share the current full source SHA256
`05e3c3998412945002eb4bf46d6e1ddf51ff8f5f37808cda88f5cf367ac55f65`.
The latest evaluated source is archived v13, not these files. v14 has zero inference
cases. API 429 `insufficient_quota` / `credit_balance_exhausted` prevents its next
full evaluation. New ablation/LOTO and 606 runs have not been started.

## One behavioral change after v13

Three saved redundancy models omit all shipment/opening links, although the question
defines whether each supplier is operational. Their generated programs permit goods
to be supplied by closed suppliers and obtain the wrong objective. The common prefix
contains the restriction: "Only use activation-conditioned bounds when the current
query explicitly makes them conditional". This conflicts with the retained instruction
to preserve logical links required for operating/open decisions. A missing physical
capacity field does not eliminate the operational link.

v14 deletes that one duplicate prefix statement from all five notebook sources.
It retains the existing guidance to apply unconditional limits unconditionally,
retain required operating links, and derive valid bounds from current data.
No new example, question ID, objective value, coefficient or mathematical correction
is introduced. Version paths and the four study base-source hashes are updated.
The conflict and saved model omissions are confirmed; the isolated accuracy effect
of the deletion is unverified. Evidence: [saved models and conflicting instruction](activation_prompt_conflict_v13.json).

## Preserved architecture and experiment definitions

CSV modeling retains ZERO_SHOT_REACT_DESCRIPTION: Thought, CSVQA Action, Observation,
and the first protocol-valid Final Answer. Planned mode describes CSVQA's declarative
data extraction, not a replacement of the modeling agent. All six CSV routes share
this mechanism. Multiple calls for different source files are retained and their
payload tables are merged before code generation.

Route retrieval limits remain AP/FLP/NRM: 1; RA/TP/Others: 3. These are example limits,
not current entity counts. Mixture uses the Others workflow. The exact original query
controls domains and additional constraints. Full/RAG/LOTO code generation receives
exact structured CSVQA records; execution uses source-backed DataFrames. Few-shot
Only supplies complete Python-read data to modeling but excludes its numeric
Observation from code generation, which receives the symbolic model and paths/headers.

RAG Only removes all 15 effective modeling/Observation/code reference rows and fixed
query-only demonstrations, retaining CSVQA, profiling/planning/filtering/validation,
raw-source fallback and ORLM_QA retrieval. [Removal inventory](../minimal_revision_20261005/rag_only_removed_example_candidates.json).
Few-shot Only retains examples and removes CSVQA/planning, with complete unchanged
Python-read source data as the modeling Observation. Both reuse only the eventual
validated full classification cache. No v13 cache is used as a v14 study baseline.
LOTO Examples Only removes the target type from classification and model/code references
but permits every route. LOTO Examples And Route additionally bans the target workflow,
reclassifies without a gold replacement route, and guards before modeling. Mixture and
Others share the Others workflow. The original-label accuracy is zero by design when
that workflow is banned. No new study has run, so no v14 study accuracy is claimed.
Detailed route and study descriptions: [v13 reviewed implementations](CODE_REVIEW_V13.md).

## Validation and switches

Offline validation passes: 5 shared interface regressions, 35 ReAct/protocol checks,
6 source-fragment checks and 5 scalar-filter checks. No API inference is performed.
The LOTO exclusion/classification/guard functions are unchanged from the 14 passing
v13 fold checks; source inheritance is verified. All five notebooks parse, contain
English prose and have zero saved outputs. All 1,133 input hashes equal v7 and v13.
The preserved five direct-call notebooks remain byte-identical to their backups.
The Others-without-CSV formulation function is unchanged. Its live API performance
has not been tested by the CSV benchmarks; common code guidance still applies.

Full switches RUN_AUTOMATIC, RUN_FORCED_ROUTES, RUN_OTHER_DATASET, RUN_SINGLE_CASE,
RUN_VARIANTS and RUN_REDUNDANT_COLUMNS are false. Ablation RUN_API and LOTO RUN_LOTO
are false. The external runner rejects new study inference until the full evaluation
passes the user-confirmed overall gate and necessary floors. 606 remains off.

The fixed model, sampling, solver settings, 1,800-second deadline and matching
tolerances are unchanged. No repair or outcome retry is added. User-authorized
ReAct protocol restarts remain permitted and separately logged. A single successful
pass would not prove cross-round stability or guarantee unseen-data accuracy.
