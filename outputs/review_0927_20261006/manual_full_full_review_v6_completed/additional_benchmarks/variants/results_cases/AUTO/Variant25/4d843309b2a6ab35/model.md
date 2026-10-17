##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents.

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

(Each project is assigned to exactly one eligible worker.)

###### 2. Worker Assignment

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

(Each worker is assigned to at most one project.)

###### 3. Eligibility Mask

For each $(w,p)$ pair, $x_{w,p}$ is allowed only if all of the following hold:
- Worker $w$ is not on leave ($\text{on\_leave}_w = 0$)
- Worker $w$'s skill $\geq$ required skill for project $p$ (with Junior < Intermediate < Senior < Expert)
- The cost $c_{w,p}$ is nonblank (assignment is allowed only if a cost is given)

For all ineligible $(w,p)$ pairs, enforce:

$x_{w,p} = 0$

###### 4. Variable Domain

$x_{w,p} \in \{0,1\}$ for all $w \in W$, $p \in P$

---

##### Retrieved Information

###### Workers (with skill and on_leave status):

| worker_id | skill        | on_leave |
|-----------|--------------|----------|
| W12       | Junior       | 0        |
| W06       | Expert       | 0        |
| W04       | Intermediate | 0        |
| W01       | Junior       | 1        |
| W11       | Junior       | 0        |
| W10       | Intermediate | 0        |
| W02       | Intermediate | 0        |
| W00       | Expert       | 0        |

Eligible workers (on_leave=0): W12, W06, W04, W11, W10, W02, W00

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

###### Cost Matrix (USD cents):

| worker_id | P00  | P01  | P02  | P03  | P04  | P05  | skill        | on_leave |
|-----------|------|------|------|------|------|------|--------------|----------|
| W12       | 102  | 353  | 651  | 102  |      | 7    | Junior       | 0        |
| W06       |      | 822  |      | 223  | 1155 | 1055 | Expert       | 0        |
| W04       | 642  |      | 1130 | 133  | 199  | 311  | Intermediate | 0        |
| W11       | 1091 | 324  | 379  | 272  |      |      | Junior       | 0        |
| W10       | 1176 | 1111 | 1380 | 542  | 158  | 922  | Intermediate | 0        |
| W02       |      | 1397 | 953  | 714  | 205  |      | Intermediate | 0        |
| W00       | 1063 | 219  |      | 1329 |      | 436  | Expert       | 0        |

###### Eligibility Mask (for each worker-project pair):

- Exclude W01 (on_leave=1) from all assignments.
- For each remaining worker-project pair:
    - Only allow assignment if:
        - cost is nonblank
        - worker's skill $\geq$ required_skill for the project

Skill levels (for comparison):
- Junior = 1
- Intermediate = 2
- Senior = 3
- Expert = 4

Project required_skill:
- P00: Junior (1)
- P01: Junior (1)
- P02: Junior (1)
- P03: Junior (1)
- P04: Junior (1)
- P05: Intermediate (2)

Worker skills:
- W12: Junior (1)
- W06: Expert (4)
- W04: Intermediate (2)
- W11: Junior (1)
- W10: Intermediate (2)
- W02: Intermediate (2)
- W00: Expert (4)

Eligibility by project:

- P00 (Junior): All eligible workers with nonblank cost for P00
- P01 (Junior): All eligible workers with nonblank cost for P01
- P02 (Junior): All eligible workers with nonblank cost for P02
- P03 (Junior): All eligible workers with nonblank cost for P03
- P04 (Junior): All eligible workers with nonblank cost for P04
- P05 (Intermediate): Only workers with skill $\geq$ Intermediate and nonblank cost for P05

Eligible assignment pairs (where $x_{w,p}$ may be 0 or 1):

| worker_id | P00 | P01 | P02 | P03 | P04 | P05 |
|-----------|-----|-----|-----|-----|-----|-----|
| W12       | 1   | 1   | 1   | 1   | 0   | 0   |
| W06       | 0   | 1   | 0   | 1   | 1   | 1   |
| W04       | 1   | 0   | 1   | 1   | 1   | 1   |
| W11       | 1   | 1   | 1   | 1   | 0   | 0   |
| W10       | 1   | 1   | 1   | 1   | 1   | 1   |
| W02       | 0   | 1   | 1   | 1   | 1   | 0   |
| W00       | 1   | 1   | 0   | 1   | 0   | 1   |

(1 = eligible, 0 = ineligible; blank cost or insufficient skill)

For all ineligible pairs, enforce $x_{w,p} = 0$.

##### Full Mathematical Model

Let $W = \{\text{W12}, \text{W06}, \text{W04}, \text{W11}, \text{W10}, \text{W02}, \text{W00}\}$

Let $P = \{\text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}\}$

Let $c_{w,p}$ be the cost in USD cents as given above.

Let $E = \{(w,p):$ assignment is eligible as per the mask above$\}$

$\min \sum_{(w,p) \in E} c_{w,p} x_{w,p}$

Subject to:

$\sum_{w: (w,p) \in E} x_{w,p} = 1 \quad \forall p \in P$

$\sum_{p: (w,p) \in E} x_{w,p} \leq 1 \quad \forall w \in W$

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in E$

$x_{w,p} = 0 \quad \forall (w,p) \notin E$

##### Data

{
  "workers": {
    "W12": {"skill": "Junior", "on_leave": 0},
    "W06": {"skill": "Expert", "on_leave": 0},
    "W04": {"skill": "Intermediate", "on_leave": 0},
    "W11": {"skill": "Junior", "on_leave": 0},
    "W10": {"skill": "Intermediate", "on_leave": 0},
    "W02": {"skill": "Intermediate", "on_leave": 0},
    "W00": {"skill": "Expert", "on_leave": 0}
  },
  "projects": {
    "P00": {"required_skill": "Junior"},
    "P01": {"required_skill": "Junior"},
    "P02": {"required_skill": "Junior"},
    "P03": {"required_skill": "Junior"},
    "P04": {"required_skill": "Junior"},
    "P05": {"required_skill": "Intermediate"}
  },
  "cost": {
    "W12": {"P00": 102, "P01": 353, "P02": 651, "P03": 102, "P05": 7},
    "W06": {"P01": 822, "P03": 223, "P04": 1155, "P05": 1055},
    "W04": {"P00": 642, "P02": 1130, "P03": 133, "P04": 199, "P05": 311},
    "W11": {"P00": 1091, "P01": 324, "P02": 379, "P03": 272},
    "W10": {"P00": 1176, "P01": 1111, "P02": 1380, "P03": 542, "P04": 158, "P05": 922},
    "W02": {"P01": 1397, "P02": 953, "P03": 714, "P04": 205},
    "W00": {"P00": 1063, "P01": 219, "P03": 1329, "P05": 436}
  },
  "eligibility_mask": {
    "W12": {"P00": 1, "P01": 1, "P02": 1, "P03": 1, "P04": 0, "P05": 0},
    "W06": {"P00": 0, "P01": 1, "P02": 0, "P03": 1, "P04": 1, "P05": 1},
    "W04": {"P00": 1, "P01": 0, "P02": 1, "P03": 1, "P04": 1, "P05": 1},
    "W11": {"P00": 1, "P01": 1, "P02": 1, "P03": 1, "P04": 0, "P05": 0},
    "W10": {"P00": 1, "P01": 1, "P02": 1, "P03": 1, "P04": 1, "P05": 1},
    "W02": {"P00": 0, "P01": 1, "P02": 1, "P03": 1, "P04": 1, "P05": 0},
    "W00": {"P00": 1, "P01": 1, "P02": 0, "P03": 1, "P04": 0, "P05": 1}
  }
}

All costs are in USD cents.