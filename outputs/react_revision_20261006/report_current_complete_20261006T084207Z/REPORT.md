# Current ReAct results

Full, RAG Only and LOTO Examples And Route use preserved v7 results. Few-shot Only uses the new direct-model revision, evaluated once. LOTO Examples Only has no completed v7 evaluation. Earlier Few-shot Only 63/101 is historical. No inference was performed for this report.

Classification, optimal solve and Objective Match are distinct; failures remain in denominators. Objective abs/rel tolerances and solver MIPGap are 1e-4, with the common 1800-second case deadline.

| experiment | dataset | cases | classification_correct | solved | objective_match | objective_match_percent |
| --- | --- | --- | --- | --- | --- | --- |
| Full ReAct v7 | automatic | 101 | 94 | 96 | 94 | 93.06930693069307 |
| Full ReAct v7 | variants | 36 | 32 | 34 | 34 | 94.44444444444444 |
| Full ReAct v7 | columns/50pct-S1 | 35 | 33 | 33 | 33 | 94.28571428571428 |
| Full ReAct v7 | columns/50pct-S2 | 35 | 33 | 33 | 33 | 94.28571428571428 |
| Full ReAct v7 | columns/50pct-S3 | 35 | 33 | 33 | 33 | 94.28571428571428 |
| Full ReAct v7 | columns/100pct-S1 | 35 | 33 | 32 | 32 | 91.42857142857144 |
| Full ReAct v7 | columns/100pct-S2 | 35 | 33 | 32 | 30 | 85.71428571428571 |
| Full ReAct v7 | columns/100pct-S3 | 35 | 33 | 33 | 32 | 91.42857142857144 |
| Full ReAct v7 | columns/200pct-S1 | 35 | 33 | 32 | 32 | 91.42857142857144 |
| Full ReAct v7 | columns/200pct-S2 | 35 | 33 | 33 | 33 | 94.28571428571428 |
| Full ReAct v7 | columns/200pct-S3 | 35 | 33 | 33 | 31 | 88.57142857142857 |
| RAG Only v7 | automatic | 101 | 94 | 90 | 86 | 85.14851485148515 |
| Few-shot Only direct model | automatic | 101 | 94 | 91 | 86 | 85.14851485148515 |
| LOTO Examples And Route v7 | automatic | 101 | 0 | 91 | 88 | 87.12871287128714 |

## Full ReAct v7: automatic

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 5 | 5 | 5 | 5 |
| FLP | 14 | 14 | 13 | 13 |
| NRM | 25 | 25 | 25 | 25 |
| RA | 22 | 22 | 22 | 22 |
| TP | 9 | 9 | 9 | 9 |
| Others | 8 | 7 | 6 | 5 |
| Mixture | 18 | 12 | 16 | 15 |

## Full ReAct v7: variants

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 3 | 3 | 3 | 3 |
| FLP | 3 | 3 | 3 | 3 |
| Others | 13 | 12 | 12 | 12 |
| Mixture | 17 | 14 | 16 | 16 |

## Full ReAct v7: columns/50pct-S1

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 6 | 6 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 5 | 5 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 4 | 4 |

## Full ReAct v7: columns/50pct-S2

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 6 | 6 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 6 | 6 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 3 | 3 |

## Full ReAct v7: columns/50pct-S3

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 6 | 6 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 5 | 5 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 4 | 4 |

## Full ReAct v7: columns/100pct-S1

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 6 | 6 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 4 | 4 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 4 | 4 |

## Full ReAct v7: columns/100pct-S2

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 6 | 5 |
| RA | 11 | 11 | 11 | 10 |
| TP | 6 | 6 | 5 | 5 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 3 | 3 |

## Full ReAct v7: columns/100pct-S3

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 5 | 4 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 6 | 6 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 4 | 4 |

## Full ReAct v7: columns/200pct-S1

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 5 | 5 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 6 | 6 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 3 | 3 |

## Full ReAct v7: columns/200pct-S2

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 5 | 5 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 6 | 6 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 4 | 4 |

## Full ReAct v7: columns/200pct-S3

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 4 | 4 | 4 | 4 |
| FLP | 7 | 7 | 6 | 4 |
| RA | 11 | 11 | 11 | 11 |
| TP | 6 | 6 | 6 | 6 |
| Others | 3 | 3 | 3 | 3 |
| Mixture | 4 | 2 | 3 | 3 |

## RAG Only v7: automatic

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 5 | 5 | 5 | 5 |
| FLP | 14 | 14 | 13 | 11 |
| NRM | 25 | 25 | 21 | 21 |
| RA | 22 | 22 | 19 | 18 |
| TP | 9 | 9 | 7 | 7 |
| Others | 8 | 7 | 8 | 7 |
| Mixture | 18 | 12 | 17 | 17 |

## Few-shot Only direct model: automatic

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 5 | 5 | 5 | 5 |
| FLP | 14 | 14 | 12 | 12 |
| NRM | 25 | 25 | 23 | 21 |
| RA | 22 | 22 | 22 | 22 |
| TP | 9 | 9 | 8 | 8 |
| Others | 8 | 7 | 7 | 6 |
| Mixture | 18 | 12 | 14 | 12 |

## LOTO Examples And Route v7: automatic

| category | cases | classification_correct | solved | objective_match |
| --- | --- | --- | --- | --- |
| AP | 5 | 0 | 5 | 5 |
| FLP | 14 | 0 | 13 | 13 |
| NRM | 25 | 0 | 23 | 23 |
| RA | 22 | 0 | 22 | 21 |
| TP | 9 | 0 | 8 | 8 |
| Others | 8 | 0 | 7 | 6 |
| Mixture | 18 | 0 | 13 | 12 |

## Source paths

- Full ReAct v7: `/Users/cora/Documents/GitHub/lean-llm-opt/outputs/react_revision_20261006/full_v7/case_summary.csv`
- RAG Only v7: `/Users/cora/Documents/GitHub/lean-llm-opt/outputs/react_revision_20261006/rag_only_v7/case_summary.csv`
- Few-shot Only direct model: `/Users/cora/Documents/GitHub/lean-llm-opt/outputs/react_revision_20261006/few_shot_only_v7_direct_model_20261006T071958Z/case_summary.csv`
- LOTO Examples And Route v7: `/Users/cora/Documents/GitHub/lean-llm-opt/outputs/react_revision_20261006/examples_and_route_v7/case_summary.csv`

Full covers all 452 cases. Completed studies cover 101 only. LOTO Examples Only is not run. 606 is disabled and unexecuted.