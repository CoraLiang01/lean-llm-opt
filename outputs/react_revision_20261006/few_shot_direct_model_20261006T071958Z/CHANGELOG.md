# Few-shot Only: direct formulation handoff from ReAct v7

This revision implements the user's corrected ablation boundary. At creation, this revision had not been evaluated with the API. The requested complete 101-case evaluation is now finished: classification 94/101, optimal solve 91/101, Objective Match 86/101 (85.15%). The historical v7 Few-shot Only score of 63/101 remains unchanged. All cases were run once, with failures counted in the denominator; the new score is not selected from multiple passes.

- Replace current-case CSVQA retrieval and extraction with complete original CSV fields read by Python as text, preserving files, columns, rows, order, identifiers and empty values. Stable source table identifiers are metadata only.
- Remove the symbolic-section filter. Pass the returned formulation unchanged, including parameter definitions and Data Mapping, to the code-generation LLM. Do not separately pass the current Observation. Source paths and column names remain available for runtime CSV reading.
- Use the full v7 common modeling guidance and final-answer envelope. Any shared v7 symbolic-model instructions are inherited from the full notebook; no additional Few-shot Only content restriction is imposed.
- Adapt only current-case tool/data-source instructions in the five canonical route prompts. Historical few-shot examples remain unchanged. Retain the original ReAct agent/executor, empty current-case tool list, shared protocol policy, model configuration, solver settings and matching tolerances.
- The Others CSV model is also forwarded unchanged; its code-generation stage receives source schema without rows. The Others without CSV implementation remains identical to full v7.
- Reuse the validated full v7 classification cache. No full-model model/solution is reused. RUN_API remains false and any future results use a separate directory.

Before editing, the original Few-shot Only notebook was copied into this directory. manifest.json records pre/post hashes and the new results directory. The full v7, RAG Only, LOTO and preserved direct-call notebooks were not modified.

Reproduction from the project root using the existing project Python environment:

```bash
/opt/miniconda3/envs/lean_llm_opt_4_1/bin/python outputs/react_revision_20261006/few_shot_direct_model_20261006T071958Z/apply_revision.py
/opt/miniconda3/envs/lean_llm_opt_4_1/bin/python outputs/react_revision_20261006/few_shot_direct_model_20261006T071958Z/verify_revision.py
```

Offline checks verify complete input preservation, real ReAct prompt construction without tools, absence of contradictory current-case CSVQA instructions, unchanged full-model examples and shared functions, unchanged no-CSV handling, raw formulation handoff for the 12 previously rejected models, and absence of separate source Observation in code-generation inputs. These checks do not establish generated mathematical correctness, solver success or Objective Match. No model repair, scored retry, fallback or API call was performed.

The completed pass recorded 0 model/code repairs, 0 pipeline retries, 0 protocol restarts, 0 data fallbacks and 5 SDK HTTP retries. All 12 previous transfer rejections now match. Compared with the previous pass, 26 cases recover and 3 regress (OR-024, OR-080, OR-090). The new report is in ../few_shot_only_v7_direct_model_20261006T071958Z/report/.
