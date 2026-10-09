##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where $x_{wp} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{wp}$ is the assignment cost in USD cents (only for eligible and allowed assignments).

##### Constraints

###### 1. Each project is assigned to exactly one eligible worker:

$\sum_{w \in W_{eligible}(p)} x_{wp} = 1 \quad \forall p \in P$

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P_{eligible}(w)} x_{wp} \leq 1 \quad \forall w \in W_{active}$

###### 3. Assignment only allowed if:
- Worker is not on leave,
- Worker skill $\geq$ required_skill for the project (with Junior < Intermediate < Senior < Expert),
- The cost cell is non-blank.

So, $x_{wp}$ is only defined for such $(w,p)$ pairs; for all other pairs, $x_{wp} = 0$.

###### 4. Variable domains:

$x_{wp} \in \{0,1\}$ for all eligible $(w,p)$ pairs.

---

##### Retrieved Information

###### Workers (with skill and on_leave):

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W12       | Junior      | 0        |
| W06       | Expert      | 0        |
| W04       | Intermediate| 0        |
| W01       | Junior      | 1        |
| W11       | Junior      | 0        |
| W10       | Intermediate| 0        |
| W02       | Intermediate| 0        |
| W00       | Expert      | 0        |

Only workers with on_leave = 0 are eligible:
- W12, W06, W04, W11, W10, W02, W00

###### Projects (with required_skill):

| project_id | required_skill |
|------------|---------------|
| P00        | Junior        |
| P01        | Junior        |
| P02        | Junior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Intermediate  |

###### Skill hierarchy:

Junior < Intermediate < Senior < Expert

###### Cost Matrix (USD cents, blank = forbidden):

| worker_id | P00  | P01  | P02  | P03  | P04  | P05  |
|-----------|------|------|------|------|------|------|
| W12       | 102  | 353  | 651  | 102  |      | 7    |
| W06       |      | 822  |      | 223  | 1155 | 1055 |
| W04       | 642  |      | 1130 | 133  | 199  | 311  |
| W11       | 1091 | 324  | 379  | 272  |      |      |
| W10       | 1176 | 1111 | 1380 | 542  | 158  | 922  |
| W02       |      | 1397 | 953  | 714  | 205  |      |
| W00       | 1063 | 219  |      | 1329 |      | 436  |

###### Eligibility Table (for each worker-project pair):

- Exclude W01 (on_leave=1).
- For each project, only allow assignments where:
    - The cost cell is non-blank,
    - The worker's skill is at least the required_skill.

| worker_id | skill        | P00 | P01 | P02 | P03 | P04 | P05 |
|-----------|-------------|-----|-----|-----|-----|-----|-----|
| W12       | Junior      | 102 | 353 | 651 | 102 |     |     |
| W06       | Expert      |     | 822 |     | 223 |1155 |1055 |
| W04       | Intermediate| 642 |     |1130 | 133 | 199 | 311 |
| W11       | Junior      |1091 | 324 | 379 | 272 |     |     |
| W10       | Intermediate|1176 |1111 |1380 | 542 | 158 | 922 |
| W02       | Intermediate|     |1397 | 953 | 714 | 205 |     |
| W00       | Expert      |1063 | 219 |     |1329 |     | 436 |

- For P05 (required_skill=Intermediate): only Intermediate, Senior, Expert can be assigned (W12, W11 excluded).
- For other projects (required_skill=Junior): all skills allowed.

###### Final Eligible Assignment Cost Table:

| worker_id | P00 | P01 | P02 | P03 | P04 | P05 |
|-----------|-----|-----|-----|-----|-----|-----|
| W12       | 102 | 353 | 651 | 102 |     |     |
| W06       |     | 822 |     | 223 |1155 |1055 |
| W04       | 642 |     |1130 | 133 | 199 | 311 |
| W11       |1091 | 324 | 379 | 272 |     |     |
| W10       |1176 |1111 |1380 | 542 | 158 | 922 |
| W02       |     |1397 | 953 | 714 | 205 |     |
| W00       |1063 | 219 |     |1329 |     | 436 |

- Blank cells indicate forbidden assignments.

###### Sets:

- $W = \{$W12, W06, W04, W11, W10, W02, W00$\}$
- $P = \{$P00, P01, P02, P03, P04, P05$\}$

###### Variables:

- $x_{wp} \in \{0,1\}$ for each eligible $(w,p)$ pair (see table above).

---

##### Full Mathematical Model

$\boxed{
\begin{align*}
\min \quad & \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp} \\
\text{s.t.} \quad & \sum_{w \in W_{eligible}(p)} x_{wp} = 1 \quad \forall p \in P \\
& \sum_{p \in P_{eligible}(w)} x_{wp} \leq 1 \quad \forall w \in W \\
& x_{wp} \in \{0,1\} \quad \forall (w,p) \text{ eligible} \\
& x_{wp} = 0 \quad \forall (w,p) \text{ not eligible (blank or skill/on_leave)}
\end{align*}
}$

where $c_{wp}$ and eligibility are as in the table above.

All costs are in USD cents.