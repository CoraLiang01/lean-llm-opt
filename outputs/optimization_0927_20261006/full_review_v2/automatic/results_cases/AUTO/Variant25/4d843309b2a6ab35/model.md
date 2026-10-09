##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where $x_{wp} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{wp}$ is the assignment cost in USD cents (only for allowed assignments).

##### Constraints

###### 1. Each project is assigned to exactly one eligible worker:

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

###### 3. Assignment eligibility:

- $x_{wp} = 0$ if:
    - Worker $w$ is on leave,
    - Worker $w$ does not meet the required skill for project $p$ (skill hierarchy: Junior < Intermediate < Senior < Expert; worker's skill must be $\geq$ required_skill),
    - The cost cell $c_{wp}$ is blank (forbidden assignment).

###### 4. Variable domain:

$x_{wp} \in \{0,1\}$ for all $w \in W$, $p \in P$

---

##### Retrieved Information

###### Workers (excluding on_leave=1):

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W12       | Junior      | 0        |
| W06       | Expert      | 0        |
| W04       | Intermediate| 0        |
| W11       | Junior      | 0        |
| W10       | Intermediate| 0        |
| W02       | Intermediate| 0        |
| W00       | Expert      | 0        |

###### Projects and required skills:

| project_id | required_skill |
|------------|---------------|
| P00        | Junior        |
| P01        | Junior        |
| P02        | Junior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Intermediate  |

###### Skill hierarchy (for eligibility):

Junior < Intermediate < Senior < Expert

###### Cost matrix (USD cents, blank = forbidden):

| worker_id | P00  | P01  | P02  | P03  | P04  | P05  | skill        |
|-----------|------|------|------|------|------|------|--------------|
| W12       | 102  | 353  | 651  | 102  |      | 7    | Junior       |
| W06       |      | 822  |      | 223  | 1155 | 1055 | Expert       |
| W04       | 642  |      | 1130 | 133  | 199  | 311  | Intermediate |
| W11       | 1091 | 324  | 379  | 272  |      |      | Junior       |
| W10       | 1176 | 1111 | 1380 | 542  | 158  | 922  | Intermediate |
| W02       |      | 1397 | 953  | 714  | 205  |      | Intermediate |
| W00       | 1063 | 219  |      | 1329 |      | 436  | Expert       |

###### Assignment eligibility (by skill):

- For P00–P04 (required_skill = Junior): All listed workers are eligible.
- For P05 (required_skill = Intermediate): Only workers with skill $\geq$ Intermediate (i.e., W06, W04, W10, W02, W00) are eligible. W12 and W11 (Junior) are not eligible for P05.

###### Final allowed assignments (where $x_{wp}$ can be 1):

| worker_id | P00  | P01  | P02  | P03  | P04  | P05  |
|-----------|------|------|------|------|------|------|
| W12       | 102  | 353  | 651  | 102  |      |      |
| W06       |      | 822  |      | 223  | 1155 | 1055 |
| W04       | 642  |      | 1130 | 133  | 199  | 311  |
| W11       | 1091 | 324  | 379  | 272  |      |      |
| W10       | 1176 | 1111 | 1380 | 542  | 158  | 922  |
| W02       |      | 1397 | 953  | 714  | 205  |      |
| W00       | 1063 | 219  |      | 1329 |      | 436  |

- Blank cells indicate forbidden assignments (either by cost matrix, skill, or on_leave).

###### Sets:

- $W = \{$W12, W06, W04, W11, W10, W02, W00$\}$
- $P = \{$P00, P01, P02, P03, P04, P05$\}$

###### Cost coefficients $c_{wp}$ (USD cents):

{
  "W12": {"P00": 102, "P01": 353, "P02": 651, "P03": 102},
  "W06": {"P01": 822, "P03": 223, "P04": 1155, "P05": 1055},
  "W04": {"P00": 642, "P02": 1130, "P03": 133, "P04": 199, "P05": 311},
  "W11": {"P00": 1091, "P01": 324, "P02": 379, "P03": 272},
  "W10": {"P00": 1176, "P01": 1111, "P02": 1380, "P03": 542, "P04": 158, "P05": 922},
  "W02": {"P01": 1397, "P02": 953, "P03": 714, "P04": 205},
  "W00": {"P00": 1063, "P01": 219, "P03": 1329, "P05": 436}
}

###### Variables:

- $x_{wp} \in \{0,1\}$ for each allowed $(w,p)$ pair as above.

---

##### Complete Mathematical Model

$\boxed{
\begin{align*}
\min \quad & \sum_{w \in W} \sum_{p \in P} c_{wp} x_{wp} \\
\text{s.t.} \quad & \sum_{w \in W: (w,p) \text{ allowed}} x_{wp} = 1 \quad \forall p \in P \\
& \sum_{p \in P: (w,p) \text{ allowed}} x_{wp} \leq 1 \quad \forall w \in W \\
& x_{wp} = 0 \quad \text{if assignment forbidden (on leave, skill, or blank cell)} \\
& x_{wp} \in \{0,1\} \quad \forall (w,p) \text{ allowed}
\end{align*}
}$

All costs are in USD cents. The minimum total assignment cost is the optimal value of the above objective.