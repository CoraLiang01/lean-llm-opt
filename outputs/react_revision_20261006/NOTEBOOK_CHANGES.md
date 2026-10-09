# Restore the original ReAct architecture

The evaluated direct-call family has been restored under its original filenames.
The five current ReAct notebooks use a consistent _ReAct.ipynb suffix. See
[the notebook family index](NOTEBOOK_FAMILIES.md) and version_layout_20261006.json
for the exact mapping and hashes. The separation changed only filename/reference
assignments in the ReAct notebooks; all inference function definitions are retained.

The five current notebooks are backed up in backups_direct_call_version before edits.
Earlier direct-call scores remain unchanged under outputs/minimal_revision_20261005.
Those scores do not establish performance for this restored architecture.

NRM, RA, TP, AP and FLP use the original ZERO_SHOT_REACT_DESCRIPTION agent with CSVQA.
The LLM chooses its Action, receives the current Observation and returns a Final Answer.
Python does not call CSVQA before the agent. Under the current user policy, the
agent must execute CSVQA at least once; multiple calls may read different files using
zero-based original source indices. Missing CSVQA calls and malformed ReAct output
restart the agent within the original 1,800-second case deadline. Intermediate protocol
attempts are logged but not scored as failed cases. The first protocol-valid completion
is used. Code/solver repair, objective-based retries and truncation regeneration remain disabled.
The v1 diagnostic pass exposed final models lacking the literal Final Answer label.
It was stopped after that repeated failure; all submitted attempts finish and are retained.
The v2 prompt explicitly required the output envelope but exposed copied protocol text in Action Input and a skipped current-data tool. Its partial pass is preserved. The v3 prompt adds clear original-query boundaries and actual current-data readiness metadata. The agent still chooses its Action; the program does not force or pre-execute it. No parser salvage is added.

The v3 automatic 101 pass finished at 97 Objective Matches, but five observed Variants
failures made its 32/36 gate unreachable. Further submissions were stopped and all
started cases were allowed to finish. The v4 revision adds only three shared code
generation instructions based on those saved failures: numeric bounds instead of
None, distinct decision-variable and loop names, and comparisons used as constraints
instead of expression terms. Generated programs are never patched or regenerated.
The v4 version is evaluated afresh; successful v3 outputs are not reused.
The v4 diagnostic pass was stopped after the user changed the protocol retry policy.
Its submitted cases are preserved. The v5 pass uses a new frozen notebook and result
directory, with no selection among case outcomes from different versions.

The v5 failures confirmed additional generic generated-code problems: default CSV
parsing converted empty fields to NaN, scalar quicksum(0) was invalid, and loop names
overwrote decision containers. The v6 instructions preserve string fields and empty
values, require iterable sums and distinct decision-container names, and distinguish
unconditional bounds from activation constraints. These are shared instructions;
generated programs are never edited after generation.

The v6 diagnostic pass also exposed code treating a table object as a source record.
The current v7 transfers planned CSVQA records through CSVQA_FRAMES[table_id], a
Python-created DataFrame dictionary with the original strings, columns, row order
and source indices. CSVQA_DATA remains available for metadata. This interface applies
to planned canonical CSV routes; Few-shot Only still passes complete data only to
modeling and reads source CSVs at execution. No coefficients or dimensions are
changed. v7 also restores a cached agent's original prompt after each invocation and
logs/restarts malformed JSON CSVQA Action Input as a protocol error.

The v7 offline checks passed 35 checks with zero API calls. A new frozen v7 evaluation
has started. Its accuracy is unverified until the requested datasets finish. v5/v6
diagnostic histories are retained, including cases still running when dispatch stops.

Planned source extraction, identifier/axis validation, full-source fallback, the shared
Gurobi interface and immutable failed-case cache remain. Others with CSV retains its
original Abstract Model Plan / runtime CSV workflow. Others without CSV remains ReAct.

RAG Only removes the same modeling/code demonstrations and retains CSVQA/its plan.
Few-shot Only retains demonstrations, has no CSVQA/planner, and receives all original
CSV fields as a Python Observation. Its ReAct agent has an empty tool set, constructed
with ZeroShotAgent/AgentExecutor because initialize_agent rejects empty tool lists.
Only symbolic model sections and source paths/columns reach code generation.
Both ablations must reuse the new ReAct full classification cache, never the old cache.
LOTO target-reference removal and the pre-model disabled-route guard are unchanged.
Both LOTO methods reclassify with the filtered references and original ReAct protocol.

Every benchmark/606 switch remains off pending the new full 101, Variants and all nine
sheet evaluations. All methods use separate versioned result directories and fixed tolerances.
No API performance is claimed by the offline protocol tests. The 606 experiment is not run.
