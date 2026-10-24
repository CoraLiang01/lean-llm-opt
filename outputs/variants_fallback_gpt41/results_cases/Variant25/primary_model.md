##### Objective Function:

$\quad \min \sum_{i \in \mathcal{W}} \sum_{j \in \mathcal{P}} c_{ij} x_{ij}$

where $x_{ij} = 1$ if worker $i$ is assigned to project $j$, $0$ otherwise, and $c_{ij}$ is the assignment cost in USD cents.

##### Sets and Parameters

- $\mathcal{W}$: Eligible workers (not on leave, skill $\geq$ required_skill)
- $\mathcal{P}$: Projects

###### Workers (with skill and on_leave status):

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W12       | Junior      | 0        |
| W06       | Expert      | 0        |
| W04       | Intermediate| 0        |
| W01       | Junior      | 1        |  (excluded)
| W11       | Junior      | 0        |
| W10       | Intermediate| 0        |
| W02       | Intermediate| 0        |
| W00       | Expert      | 0        |

$\mathcal{W} = \{\text{W00}, \text{W02}, \text{W04}, \text{W06}, \text{W10}, \text{W11}, \text{W12}\}$

(Exclude W01: on_leave=1)

###### Projects (with required_skill):

| project_id | required_skill |
|------------|---------------|
| P00        | Junior        |
| P01        | Junior        |
| P02        | Junior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Intermediate  |

$\mathcal{P} = \{\text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}\}$

###### Skill hierarchy: Junior < Intermediate < Senior < Expert

##### Cost Matrix $c_{ij}$ (USD cents):

| worker_id | P00  | P01  | P02  | P03  | P04  | P05  | skill        |
|-----------|------|------|------|------|------|------|--------------|
| W12       | 102  | 353  | 651  | 102  |      | 7    | Junior       |
| W06       |      | 822  |      | 223  | 1155 | 1055 | Expert       |
| W04       | 642  |      | 1130 | 133  | 199  | 311  | Intermediate |
| W11       | 1091 | 324  | 379  | 272  |      |      | Junior       |
| W10       | 1176 | 1111 | 1380 | 542  | 158  | 922  | Intermediate |
| W02       |      | 1397 | 953  | 714  | 205  |      | Intermediate |
| W00       | 1063 | 219  |      | 1329 |      | 436  | Expert       |

##### Project required_skill:

| project_id | required_skill |
|------------|---------------|
| P00        | Junior        |
| P01        | Junior        |
| P02        | Junior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Intermediate  |

##### Eligibility Mask

A worker $i$ is eligible for project $j$ if:
- $i$ is not on leave,
- $i$'s skill $\geq$ required_skill of $j$,
- $c_{ij}$ is not blank.

Skill levels (ordinal): Junior=1, Intermediate=2, Senior=3, Expert=4

| worker_id | skill_level |
|-----------|-------------|
| W12       | 1           |
| W06       | 4           |
| W04       | 2           |
| W11       | 1           |
| W10       | 2           |
| W02       | 2           |
| W00       | 4           |

| project_id | required_skill_level |
|------------|---------------------|
| P00        | 1                   |
| P01        | 1                   |
| P02        | 1                   |
| P03        | 1                   |
| P04        | 1                   |
| P05        | 2                   |

Eligible pairs (where $x_{ij}$ can be $1$):

- W12: P00, P01, P02, P03, P05 (P04 blank; P05 skill=1 < 2, so ineligible)
- W06: P01, P03, P04, P05 (all skill ok, but only nonblank cells)
- W04: P00, P02, P03, P04, P05 (all skill ok, nonblank cells)
- W11: P00, P01, P02, P03 (P04, P05 blank; skill=1 < 2 for P05, so ineligible)
- W10: all projects (all skill ok, nonblank cells)
- W02: P01, P02, P03, P04 (P00, P05 blank; all skill ok)
- W00: P00, P01, P03, P05 (all skill ok, nonblank cells)

So, the eligibility mask is:

| worker_id | P00 | P01 | P02 | P03 | P04 | P05 |
|-----------|-----|-----|-----|-----|-----|-----|
| W12       | 1   | 1   | 1   | 1   | 0   | 0   |
| W06       | 0   | 1   | 0   | 1   | 1   | 1   |
| W04       | 1   | 0   | 1   | 1   | 1   | 1   |
| W11       | 1   | 1   | 1   | 1   | 0   | 0   |
| W10       | 1   | 1   | 1   | 1   | 1   | 1   |
| W02       | 0   | 1   | 1   | 1   | 1   | 0   |
| W00       | 1   | 1   | 0   | 1   | 0   | 1   |

##### Decision Variables

$x_{ij} \in \{0,1\}$ for all eligible $(i,j)$ pairs (as above). For ineligible pairs, $x_{ij} = 0$.

##### Constraints

1. **Each project assigned to exactly one worker:**

$\sum_{i \in \mathcal{W}} x_{ij} = 1 \quad \forall j \in \mathcal{P}$

2. **Each worker assigned to at most one project:**

$\sum_{j \in \mathcal{P}} x_{ij} \leq 1 \quad \forall i \in \mathcal{W}$

3. **Eligibility and forbidden assignments:**

For each $(i,j)$ pair not in the eligibility mask above, $x_{ij} = 0$.

##### Full Mathematical Model

$\min \sum_{i \in \mathcal{W}} \sum_{j \in \mathcal{P}} c_{ij} x_{ij}$

subject to:

$\sum_{i \in \mathcal{W}} x_{ij} = 1 \quad \forall j \in \mathcal{P}$

$\sum_{j \in \mathcal{P}} x_{ij} \leq 1 \quad \forall i \in \mathcal{W}$

$x_{ij} = 0$ for all ineligible $(i,j)$ pairs (see eligibility mask above)

$x_{ij} \in \{0,1\}$ for all eligible $(i,j)$ pairs

##### Retrieved Information

{
  "workers": [
    {"worker_id": "W12", "skill": "Junior", "on_leave": "0"},
    {"worker_id": "W06", "skill": "Expert", "on_leave": "0"},
    {"worker_id": "W04", "skill": "Intermediate", "on_leave": "0"},
    {"worker_id": "W11", "skill": "Junior", "on_leave": "0"},
    {"worker_id": "W10", "skill": "Intermediate", "on_leave": "0"},
    {"worker_id": "W02", "skill": "Intermediate", "on_leave": "0"},
    {"worker_id": "W00", "skill": "Expert", "on_leave": "0"}
  ],
  "projects": [
    {"project_id": "P00", "required_skill": "Junior"},
    {"project_id": "P01", "required_skill": "Junior"},
    {"project_id": "P02", "required_skill": "Junior"},
    {"project_id": "P03", "required_skill": "Junior"},
    {"project_id": "P04", "required_skill": "Junior"},
    {"project_id": "P05", "required_skill": "Intermediate"}
  ],
  "cost": {
    "W12": {"P00": 102, "P01": 353, "P02": 651, "P03": 102, "P04": null, "P05": 7},
    "W06": {"P00": null, "P01": 822, "P02": null, "P03": 223, "P04": 1155, "P05": 1055},
    "W04": {"P00": 642, "P01": null, "P02": 1130, "P03": 133, "P04": 199, "P05": 311},
    "W11": {"P00": 1091, "P01": 324, "P02": 379, "P03": 272, "P04": null, "P05": null},
    "W10": {"P00": 1176, "P01": 1111, "P02": 1380, "P03": 542, "P04": 158, "P05": 922},
    "W02": {"P00": null, "P01": 1397, "P02": 953, "P03": 714, "P04": 205, "P05": null},
    "W00": {"P00": 1063, "P01": 219, "P02": null, "P03": 1329, "P04": null, "P05": 436}
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