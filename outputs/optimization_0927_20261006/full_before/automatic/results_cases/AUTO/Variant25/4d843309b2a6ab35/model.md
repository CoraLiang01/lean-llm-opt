##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where $x_{wp} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{wp}$ is the assignment cost in USD cents (forbidden assignments have no variable and do not contribute).

##### Constraints

###### 1. Each project is assigned to exactly one eligible worker:

$\sum_{w \in W'} x_{wp} = 1 \quad \forall p \in P$

where $W'$ is the set of eligible workers (not on leave, skill $\geq$ required_skill, and $c_{wp}$ not blank).

###### 2. Each eligible worker is assigned to at most one project:

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W'$

###### 3. Assignment only allowed if:

- Worker is not on leave ($on\_leave = 0$)
- Worker skill $\geq$ project required_skill (Junior < Intermediate < Senior < Expert)
- $c_{wp}$ is not blank

If any of these is not satisfied, $x_{wp}$ is not defined (or is fixed to $0$).

###### 4. Variable constraints:

$x_{wp} \in \{0,1\}$ for all allowed $(w,p)$ pairs.

---

##### Retrieved Information

```json
{
  "workers": [
    {"worker_id": "W12", "skill": "Junior", "on_leave": 0},
    {"worker_id": "W06", "skill": "Expert", "on_leave": 0},
    {"worker_id": "W04", "skill": "Intermediate", "on_leave": 0},
    {"worker_id": "W01", "skill": "Junior", "on_leave": 1},
    {"worker_id": "W11", "skill": "Junior", "on_leave": 0},
    {"worker_id": "W10", "skill": "Intermediate", "on_leave": 0},
    {"worker_id": "W02", "skill": "Intermediate", "on_leave": 0},
    {"worker_id": "W00", "skill": "Expert", "on_leave": 0}
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
    "W12": {"P00": 102, "P01": 353, "P02": 651, "P03": 102, "P04": "",   "P05": 7},
    "W06": {"P00": "",   "P01": 822, "P02": "",   "P03": 223, "P04": 1155, "P05": 1055},
    "W04": {"P00": 642, "P01": "",   "P02": 1130, "P03": 133, "P04": 199,  "P05": 311},
    "W01": {"P00": 5,   "P01": "",   "P02": 20,   "P03": "",   "P04": 18,   "P05": 7},
    "W11": {"P00": 1091,"P01": 324, "P02": 379,  "P03": 272, "P04": "",    "P05": ""},
    "W10": {"P00": 1176,"P01": 1111,"P02": 1380, "P03": 542, "P04": 158,   "P05": 922},
    "W02": {"P00": "",  "P01": 1397,"P02": 953,  "P03": 714, "P04": 205,   "P05": ""},
    "W00": {"P00": 1063,"P01": 219, "P02": "",   "P03": 1329,"P04": "",    "P05": 436}
  }
}
```

###### Eligible workers (not on leave):

- W12 (Junior)
- W06 (Expert)
- W04 (Intermediate)
- W11 (Junior)
- W10 (Intermediate)
- W02 (Intermediate)
- W00 (Expert)

###### Projects and required skills:

- P00: Junior
- P01: Junior
- P02: Junior
- P03: Junior
- P04: Junior
- P05: Intermediate

###### Skill hierarchy: Junior < Intermediate < Senior < Expert

###### Allowed assignments (worker skill $\geq$ required_skill and cost cell not blank):

| worker_id | skill        | P00  | P01  | P02  | P03  | P04  | P05  |
|-----------|-------------|------|------|------|------|------|------|
| W12       | Junior      | 102  | 353  | 651  | 102  |      |      |
| W06       | Expert      |      | 822  |      | 223  | 1155 | 1055 |
| W04       | Intermediate| 642  |      | 1130 | 133  | 199  | 311  |
| W11       | Junior      | 1091 | 324  | 379  | 272  |      |      |
| W10       | Intermediate| 1176 | 1111 | 1380 | 542  | 158  | 922  |
| W02       | Intermediate|      | 1397 | 953  | 714  | 205  |      |
| W00       | Expert      | 1063 | 219  |      | 1329 |      | 436  |

- For P05 (Intermediate), only Intermediate, Senior, or Expert workers are eligible.

---

##### Variable definitions

Let $x_{wp} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, for all allowed $(w,p)$ pairs as above.

---

##### Model summary

- Minimize total assignment cost in USD cents.
- Each project assigned to exactly one eligible worker.
- Each eligible worker assigned to at most one project.
- Only allowed assignments (not on leave, skill $\geq$ required, cost cell not blank).
- $x_{wp} \in \{0,1\}$ for all allowed $(w,p)$.

---

##### All parameters and sets are fully specified above.