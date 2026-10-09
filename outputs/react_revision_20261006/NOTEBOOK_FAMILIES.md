# Preserved notebook families

The direct-call delivery is restored byte for byte from its pre-ReAct backup.
Its scores belong to outputs/minimal_revision_20261005. ReAct evaluation records
remain under outputs/react_revision_20261006. No results are combined across families.

| Method | Preserved direct-call notebook | Current ReAct notebook |
| --- | --- | --- |
| full | [LEAN_LLM_OPT_4.1_Large-scale.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/LEAN_LLM_OPT_4.1_Large-scale.ipynb) | [LEAN_LLM_OPT_4.1_Large-scale_1006_ReAct.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/LEAN_LLM_OPT_4.1_Large-scale_1006_ReAct.ipynb) |
| rag_only | [Ablation_Study_Large_Scale_Or_RAG_Only.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_RAG_Only.ipynb) | [Ablation_Study_Large_Scale_Or_RAG_Only_ReAct.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_RAG_Only_ReAct.ipynb) |
| few_shot_only | [Ablation_Study_Large_Scale_Or_Few-shot_Only.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_Few-shot_Only.ipynb) | [Ablation_Study_Large_Scale_Or_Few-shot_Only_ReAct.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/Ablation_Study_Large_Scale_Or_Few-shot_Only_ReAct.ipynb) |
| examples_only | [LOTO_Examples_Only_GPT4.1_Large-scale.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/LOTO_Examples_Only_GPT4.1_Large-scale.ipynb) | [LOTO_Examples_Only_GPT4.1_Large-scale_ReAct.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/LOTO_Examples_Only_GPT4.1_Large-scale_ReAct.ipynb) |
| examples_and_route | [LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/LOTO_Examples_And_Route_GPT4.1_Large-scale.ipynb) | [LOTO_Examples_And_Route_GPT4.1_Large-scale_ReAct.ipynb](/Users/cora/Documents/GitHub/lean-llm-opt/LOTO_Examples_And_Route_GPT4.1_Large-scale_ReAct.ipynb) |

Only NOTEBOOK_FILENAME and the two LOTO_BASE_NOTEBOOK filename assignments
changed in the ReAct notebooks during this separation. Modeling prompts, ReAct
agents, CSVQA, execution, tolerances and solver settings were not changed.
The classification-cache baseline hashes continue to identify the actual frozen
full_v7 evaluation, not a different model. The evaluator permits self filename
relocation while requiring all other definition cells to remain identical.

The original direct-call full model has planned canonical CSV routes and the
original legacy Others CSV flow. Few-shot Only deliberately overrides CSV modes
to direct full-source Observation. Classification and query-only Others keep their
original architectures. No mode was changed simply to relabel the preserved code.

The restored direct full notebook retains its previously enabled 606 switch;
606 was not executed. The ReAct 606 switch remains disabled pending its own gate.
Both result sets and all frozen notebook snapshots remain intact.

The full delivery differs from its evaluated snapshot only in the reviewed 606
control cell. The two direct LOTO deliveries retain the documented no-CSV alias
correction; that branch was not API-evaluated by the all-CSV campaign.
