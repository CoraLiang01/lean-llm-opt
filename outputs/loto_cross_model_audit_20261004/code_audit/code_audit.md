# LOTO notebook implementation audit — 2026-10-04

This is a read-only audit of the four current LOTO notebooks and both model-specific full notebooks. No notebook, original data, or historical result was changed. No language-model API, embedding service, Gurobi solve, or generated program was executed. Cell references below use **zero-based notebook indices**; add one for a one-based displayed cell count.

## Main findings

1. The LOTO intervention itself is aligned. Common cells 37 (fold controls) and 41 (fold runner) are byte-identical in all four notebooks. Reference removal is based on exact normalized semantic Type; route prohibition is applied before dispatch. An offline harness passed all 28 model/variant/fold combinations, including cache boundaries and OSS parser fallback prompts.
2. The two *model-specific pipelines* are materially different. They are not one identical system with only the model name changed. Data extraction, code-reference use, parser recovery, code normalization, and solver handling all differ. These differences mostly already exist in the two full notebooks and are not newly introduced by LOTO.
3. GPT-4.1's actual 19 Examples-Only `NoneType` failures are generated functions that do not return their local Gurobi model. Replacing `namespace['m']` with a more tolerant global lookup alone will not fix those 19 examples.
4. Current notebook source can be statically verified. Exact historical runtime-source reconstruction is limited: the manifests contain hashes but not complete runtime source snapshots or environment/weight archives; the local original reference CSV is currently unreadable through the audit process.

## Within-model alignment

### GPT-4.1

- Both LOTO notebooks' original core code cells 2 and 5–35 are identical to the corresponding full notebook.
- Configuration cell 3 changes result/output notebook paths and adds `LOTO_VARIANT`, `LOTO_REMOVE_ROUTE`, and base provenance metadata.
- The two GPT variants differ in the expected configuration and description; common LOTO logic is identical.
- Current full baseline SHA-256 matches its declared baseline hash exactly: `a68c6396a9a7a005267bd3d376abd55263794bb7bad611dc8c5984802ad89e80`.

### OSS-20B

- Core logic follows its own OSS full baseline.
- Cell 29 adds `pipeline_stage` labels for reporting; this does not change modeling or execution decisions.
- Examples-and-Route additionally uses `loto_allowed_labels()` in two cell-13 recovery paths: duplicate-tool parser correction and final-label completion. This is necessary to preserve route exclusion when its model-specific recovery mechanism is used.
- Current full baseline SHA-256 matches its declared baseline hash exactly: `912c7d1beca1f99cf05eb681662ea85cab61850428a4479d79577eec1df8d795`.

## Cross-model implementation differences

| Component | GPT-4.1 | OSS-20B | Interpretation |
|---|---|---|---|
| CSV extraction modes, cell 3 | NRM planned; RA/TP/AP/FLP legacy | NRM/RA/TP/AP/FLP planned | Moving a problem to another route changes different data-processing paths in the two systems. |
| Structured data binding, cells 27/29 | NRM uses CSVQA_DATA; RA legacy uses parsed LEGACY_RECORDS | All five structured routes use CSVQA_DATA | Different risks of copying, extraction, and interface errors. |
| CSV code references, cell 27 | `_generate_code()` calls `retrieve_csv_code_example()` (k=2) on the filtered same-route pool | `get_csv_code()` appends fixed `CSV_CODE_GUIDANCE`; it does not retrieve CSV code examples for the five structured routes | Removing a same-type CSV example does not remove the same source of code assistance in both systems. |
| Others code references, cell 25 | Filtered CSV examples provide both formulation and Code | Filtered CSV examples provide both formulation and Label_Code | This branch uses the filtered CSV library in both implementations. |
| Classifier, cell 13 | Default ReAct parser, maximum 4 iterations | Label-tolerant parser, maximum 6 iterations, a completion call for unparseable final output | Different inference procedures and failure/recovery opportunities. |
| Embeddings, cells 3/5 | text-embedding-ada-002 | nomic-embed-text through Ollama | The five retrieved classification examples need not be identical even with identical reference exclusions. |
| Solver extraction, cell 29 | `namespace['m'] if 'm' in namespace else namespace['model']` | Prefer m/model, then search for a unique exposed gp.Model | GPT is less tolerant of names/None values, although neither can recover a model that exists only in a finished function's locals. |
| Generated code normalization, cell 29 | Strip bulk display names | Normalize JSON-like literals, selected imports, display names, MIPGap settings, and some DataFrame index/drop patterns | OSS includes extra deterministic adapters. |
| Solver contract | Prompt requests MIPGap=1e-4 | Prompt/normalization request MIPGap=1e-6; runtime defaults to a 180-second TimeLimit | Same final objective-match tolerance does not imply identical solve conditions. |
| Saved artifact integrity, cell 33 | Per-artifact hash checks before reuse | File presence and readability checks without per-artifact hashes | Provenance strength differs; no evidence by itself of corruption. |

Both models preserve fixed prompt examples and category definitions. This is reference-library/workflow-availability leave-out, not removal of all knowledge of a mathematical family.

## The GPT-4.1 model-return failures

The saved code for all 19 Examples-Only `NoneType` failures was parsed with `ast`, not executed. In every case a global assignment has the form `m = solve_function(...)`, but that function contains no return statement for the Gurobi model. The model is local to that function. Therefore `m` becomes `None`, and `with model:` in `execute_code()` fails.

| Route | Cases |
|---|---|
| TP | OR-004, OR-006, OR-008 |
| NRM | OR-018, OR-023, OR-024, OR-026, OR-034 |
| RA | OR-036, OR-039, OR-041, OR-044, OR-046, OR-047, OR-049, OR-052, OR-053 |
| FLP | OR-063, OR-066 |

Every one of these cases used its own held-out route, whose same-type CSV code examples were absent in GPT Examples-Only. GPT Examples-and-Route has no `NoneType` failures in the saved audit. This supports an execution-interface explanation for a substantial part of the observed aggregate improvement. It does **not** establish that all 19 mathematical models were correct or would have matched the reference objective after repairing the interface.

The mock regression distinguishes four situations:

| Simulated program namespace | GPT-4.1 wrapper | OSS wrapper |
|---|---|---|
| m is a valid model | Accepted | Accepted |
| m is None, global model is valid | Fails | Accepted |
| One valid model with another global variable name | Fails | Accepted |
| m is None, no model exposed globally | Fails | Fails |

The last row matches the 19 actual saved failures. A consistent future fix must enforce or capture the generated model-return contract, not merely swap the lookup expression. Repaired execution should be reported as a separate controlled re-evaluation with the same policy applied to both variants and both models; it must not silently replace the original score.

## Control checks that passed

`offline_controls_check.py` uses actual notebook definitions with fake services and a 15-row fixture reconstructed from the saved manifests' Type/prompt fields; the non-Type reference contents are placeholders. It checks:

- All current Python code cells compile.
- Four notebooks times seven folds remove the correct semantic type and preserve the other semantic type when Mixture/Others share a physical route.
- Classification pools are refreshed and supply five examples.
- Empty same-route reference pools return no examples without building an empty index.
- Allowed labels propagate to OSS parser correction and fallback completion.
- Route guards reject forbidden routes before formulation; all permitted routes remain callable.
- Fingerprints differ across seven folds under the same mocked source and case.
- Real execution-wrapper functions have the mocked model-handle behavior above.

This is a control-flow test, not a test of real embedding rankings, original numerical CSV content, solver outcomes, or historical output authenticity.

## Remaining maintenance/provenance issues

- Current GPT notebooks have 47 and 46 cells; the OSS notebooks have 50. The previous `test_loto_notebooks.py` assumes all have 50 and indexes cell 47, so it fails before testing controls. This is a stale report-layout assertion, not evidence that route filtering is wrong.
- The GPT notebooks currently lack the extra saved-report cells present in OSS. Their common `loto_report()` still writes totals, per-type counts, selected-route counts, failures, and case summaries, but it does not itself render both full classification matrices.
- GPT run cells currently set `RUN_LOTO=True`, whereas OSS run cells default to False. Thus the GPT introduction's “inference disabled by default” statement no longer matches its run switch. This does not alter saved score calculations but matters when someone executes all cells.
- Base notebook hashes match. Per-run source fingerprints cannot be recomputed from current sources alone: they include original reference/data files, and the actual OSS run-site files and environment were not supplied as a complete snapshot.
- OSS uses a model tag, `gpt-oss:20b`, without an archived immutable weight digest in the shown manifests. Effective context/output limits, seed, package versions, and solver environment should be saved for future runs.

No historical scores were recalculated by this code-only sub-audit. The separate result audits establish counts from saved artifacts.
