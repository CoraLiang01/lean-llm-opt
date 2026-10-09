# Prompt and configuration materials for the current GPT-4.1 component experiments

This archive documents the frozen full workflow and its current RAG Only / Few-shot Only evaluations on the same 101 main benchmark instances. It does not modify any notebook or experimental result.

## Contents

- `configuration.json`: verified model, decoding, retrieval, tool-interface, failure-handling and scoring settings.
- `retrieval_settings.csv`: route-specific candidate counts and requested/effective example counts. RAG Only inserts zero formulation/code examples; Few-shot Only retains the full-workflow example selection.
- `RAG_Examples_All.csv`: the exact 15-row reference collection, verified against all three run manifests. Reference formulations/codes are training references, not benchmark reference answers.
- `automatic_classification_predictions.csv`: the full workflow predictions held fixed across the two ablations. This file contains no ground-truth labels or reference objectives.
- `full`, `rag_only`, `few_shot_only`: exact source cells from the evaluated frozen notebooks (`prompt_source.txt`), original run manifests, and 101 complete per-case chat-message logs per method (`requests/OR-xxx.jsonl`). Source includes prompt strings, dynamically appended instructions and assembly logic. It is documentation text rather than an executable replacement notebook.
- `evaluation_only/`: the three 101-row GPT-4.1 score exports, including errors and reference objectives for post-hoc evaluation only. These reference values are not added to generation prompts.
- `environment_observed.json`: the environment inventory inspected when this archive was prepared. Historical run manifests record Python; no separate historical package lock is claimed.
- `oss20b_r3/evaluation_only/scored_cases.csv`: the supplied 202-row oss-20b score export. Its missing generation/configuration metadata are explicitly listed in `oss20b_r3/scope.json`.
- `manifest.json`: source notebook fingerprints, case/request counts and SHA-256 values of every exported file.

## Reading the exact prompts

Each requests JSONL entry records a chat-model invocation, including the complete system/human/AI messages as serialized by the LangChain callback. The role strings are LangChain message types. Multi-step ReAct histories, tool observations, retrieved examples, and expanded current-instance inputs are retained rather than abbreviated. SDK HTTP retries can resend a request without creating a new chat-model-start entry; retry settings and per-case original records remain separate.

Use the exact prompt source together with the expanded message logs. The source documents template construction; the logs document the actual messages for successful and unsuccessful attempts. Neither API credentials nor HTTP authorization headers are included. Benchmark source CSV files are not bundled separately in this prompt archive; their paths and hashes are in the original run manifests.

## Important metric and interface boundaries

- The recorded GPT-4.1 executor counts Solved only when the live Gurobi model reports GRB.OPTIMAL. Objective match additionally requires finite objective agreement under relative and absolute tolerance 1e-4.
- Case errors, rejected truncation, and external timeouts remain in the denominator. Printed numerical answers cannot replace the model object.
- Both ablations reuse full-workflow classification. They measure changes after classification.
- RAG Only retains CSVQA/Python data handling and removes formulation/code examples.
- Few-shot Only retains examples and replaces current-instance CSVQA planning/extraction with complete Python-loaded observations. Raw complete observations are not directly appended to numerical-route code prompts; necessary coefficients flow through the numerical formulation. NRM no longer receives structured runtime data injection. Others retains CSV reading at execution.
- The supplied oss-20b CSV verifies scores and errors only. Its checkpoint, prompts, retrieval, decoding and retry parameters cannot be inferred from this file or from the GPT-4.1 configuration.
