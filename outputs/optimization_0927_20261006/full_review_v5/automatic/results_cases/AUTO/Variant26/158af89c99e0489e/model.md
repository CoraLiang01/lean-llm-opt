##### Objective Function:

$\quad \min \sum_{(w,p) \in \mathcal{E}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

where $\mathcal{E}$ is the set of all eligible (worker, project) pairs as listed below.

##### Constraints:

1. **Each project is assigned to exactly one worker:**

$\sum_{w: (w,p) \in \mathcal{E}} x_{w,p} = 1 \quad \forall p \in \{P00, P01, P02, P03, P04, P05, P06, P07\}$

2. **Each worker is assigned to at most one project:**

$\sum_{p: (w,p) \in \mathcal{E}} x_{w,p} \leq 1 \quad \forall w \in \{W00, W01, W04, W06, W10, W11, W14, W15, W19, W20\}$

3. **Assignment only allowed for listed, eligible offers:**

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{E}$

##### Retrieved Information

Eligible assignment pairs (offers) and their costs (in USD cents):

| worker_id | project_id | cost_cents |
|-----------|------------|------------|
| W10 | P00 | 147 |
| W10 | P01 | 114 |
| W10 | P03 | 998 |
| W10 | P04 | 1270 |
| W10 | P07 | 953 |
| W19 | P00 | 5 |
| W19 | P02 | 9 |
| W19 | P03 | 109 |
| W19 | P04 | 1306 |
| W19 | P05 | 626 |
| W19 | P06 | 466 |
| W19 | P07 | 1365 |
| W11 | P00 | 235 |
| W11 | P02 | 1034 |
| W11 | P03 | 782 |
| W11 | P04 | 556 |
| W11 | P05 | 1018 |
| W11 | P07 | 1136 |
| W00 | P01 | 17 |
| W00 | P04 | 430 |
| W00 | P05 | 1385 |
| W00 | P06 | 260 |
| W00 | P07 | 1107 |
| W20 | P00 | 675 |
| W20 | P01 | 1055 |
| W20 | P02 | 8 |
| W20 | P03 | 703 |
| W20 | P06 | 893 |
| W20 | P07 | 129 |
| W04 | P00 | 4 |
| W04 | P01 | 8 |
| W04 | P02 | 16 |
| W04 | P03 | 543 |
| W04 | P04 | 205 |
| W04 | P05 | 1149 |
| W04 | P06 | 533 |
| W04 | P07 | 732 |
| W01 | P00 | 19 |
| W01 | P02 | 6 |
| W01 | P05 | 19 |
| W01 | P06 | 8 |
| W01 | P07 | 13 |
| W06 | P00 | 1217 |
| W06 | P01 | 425 |
| W06 | P02 | 1320 |
| W06 | P03 | 1097 |
| W06 | P04 | 822 |
| W06 | P05 | 221 |
| W06 | P06 | 496 |
| W06 | P07 | 908 |
| W14 | P01 | 1361 |
| W14 | P03 | 379 |
| W14 | P04 | 239 |
| W14 | P05 | 460 |
| W14 | P06 | 130 |
| W15 | P00 | 722 |
| W15 | P02 | 577 |
| W15 | P03 | 1062 |
| W15 | P04 | 1314 |
| W15 | P05 | 528 |
| W15 | P06 | 835 |
| W15 | P07 | 918 |

- Only the above (worker, project) pairs are eligible for assignment.
- Each $x_{w,p}$ is a binary variable indicating if worker $w$ is assigned to project $p$.

##### Sets

- Workers: $\{W00, W01, W04, W06, W10, W11, W14, W15, W19, W20\}$
- Projects: $\{P00, P01, P02, P03, P04, P05, P06, P07\}$
- Eligible pairs: as listed above.

##### Variables

- $x_{w,p} \in \{0,1\}$ for each eligible $(w,p)$ pair.

##### Parameters

- $\text{cost\_cents}_{w,p}$: as listed in the table above.

##### Notes

- Only listed offers are permitted.
- Each project must be assigned to exactly one worker.
- Each worker may be assigned to at most one project.
- All assignments must respect the eligibility as defined by the offer list and any exclusion criteria (already enforced in the eligible pairs above).

##### Objective

Minimize the total cost in USD cents:

$\min \sum_{(w,p) \in \mathcal{E}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

subject to the constraints above.