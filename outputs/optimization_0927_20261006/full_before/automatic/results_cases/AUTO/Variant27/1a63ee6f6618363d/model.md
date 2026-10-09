##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where:
- $W$ = set of available workers (not on leave):  
$\{ \text{W00}, \text{W01}, \text{W02}, \text{W05}, \text{W10}, \text{W11}, \text{W14}, \text{W16}, \text{W17}, \text{W18}, \text{W24} \}$
- $P$ = set of projects:  
$\{ \text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}, \text{P06}, \text{P07}, \text{P08}, \text{P09} \}$
- $c_{wp}$ = assignment cost in USD cents for worker $w$ to project $p$ (see matrix below)
- $x_{wp} \in \{0,1\}$: 1 if worker $w$ is assigned to project $p$, 0 otherwise

##### Constraints

###### 1. Each project is assigned exactly one worker:

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

###### 3. Skill and forbidden assignment constraints:

- $x_{wp} = 0$ if:
    - $c_{wp}$ is blank (forbidden by cost matrix), or
    - worker $w$'s skill is less than project $p$'s required_skill (with Junior < Intermediate < Senior < Expert)

###### 4. Variable domain:

$x_{wp} \in \{0,1\} \quad \forall w \in W,\, p \in P$

---

##### Retrieved Information

###### Worker Data

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W00       | Intermediate| 0        |
| W01       | Intermediate| 0        |
| W02       | Junior      | 0        |
| W05       | Senior      | 0        |
| W10       | Expert      | 0        |
| W11       | Expert      | 0        |
| W14       | Expert      | 0        |
| W16       | Senior      | 0        |
| W17       | Junior      | 0        |
| W18       | Intermediate| 0        |
| W24       | Expert      | 0        |

###### Project Data

| project_id | required_skill |
|------------|---------------|
| P00        | Junior        |
| P01        | Junior        |
| P02        | Junior        |
| P03        | Intermediate  |
| P04        | Junior        |
| P05        | Junior        |
| P06        | Junior        |
| P07        | Expert        |
| P08        | Junior        |
| P09        | Intermediate  |

###### Skill Hierarchy

Junior < Intermediate < Senior < Expert

###### Cost Matrix ($c_{wp}$, in USD cents; blank = forbidden):

| worker_id | skill        | P00  | P01  | P02  | P03  | P04  | P05  | P06  | P07  | P08  | P09  |
|-----------|-------------|------|------|------|------|------|------|------|------|------|------|
| W00       | Intermediate|      | 1116 | 414  |      | 554  | 113  |      | 11   | 853  | 1142 |
| W01       | Intermediate|      | 170  |      | 919  | 660  | 1314 | 668  | 4    | 462  | 614  |
| W02       | Junior      | 330  | 193  |      | 4    | 431  |      | 1130 | 15   | 644  | 9    |
| W05       | Senior      | 325  | 772  | 1042 |      | 1394 |      | 374  | 2    | 1140 | 127  |
| W10       | Expert      | 1079 | 1128 | 758  |      |      | 108  |      | 423  | 744  | 347  |
| W11       | Expert      | 1143 | 841  | 920  | 122  | 634  |      | 1250 | 836  | 180  | 1105 |
| W14       | Expert      |      | 687  | 126  | 547  | 125  | 1358 | 1391 | 814  | 1362 | 962  |
| W16       | Senior      | 265  | 1114 |      | 1209 | 231  | 221  |      | 17   | 135  |      |
| W17       | Junior      | 538  |      | 1359 | 20   | 1366 | 702  | 122  | 13   | 358  |      |
| W18       | Intermediate| 238  |      | 1085 | 529  | 582  | 889  | 139  | 18   | 630  | 538  |
| W24       | Expert      | 784  | 1178 | 1171 |      | 103  | 137  | 832  | 1279 | 893  | 351  |

###### Skill Feasibility Table

- For each $(w,p)$, $x_{wp}$ is allowed only if:
    - $c_{wp}$ is not blank, and
    - worker's skill $\geq$ required_skill for project $p$.

---

##### Variable Definitions

- $x_{wp} = 1$ if worker $w$ is assigned to project $p$, 0 otherwise.
- $c_{wp}$ as above; blank entries or insufficient skill $\implies x_{wp}=0$.

---

##### Model Summary

$\min \sum_{w \in W} \sum_{p \in P} c_{wp} x_{wp}$

subject to:

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

$x_{wp} = 0$ if $c_{wp}$ is blank or worker $w$'s skill $<$ required_skill for $p$

$x_{wp} \in \{0,1\}$ for all $w,p$

---

All costs are in USD cents. The minimum total cost is the optimal value of the objective function.