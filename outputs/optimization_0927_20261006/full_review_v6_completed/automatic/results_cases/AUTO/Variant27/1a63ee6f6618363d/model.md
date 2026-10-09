##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \cdot x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents (only defined for eligible assignments).

##### Constraints

###### 1. Each project is assigned exactly one worker:

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Assignment eligibility:

- $x_{w,p} = 0$ if:
    - worker $w$ is on leave ($on\_leave=1$), or
    - $c_{w,p}$ is blank (assignment forbidden), or
    - $skill(w) < required\_skill(p)$ (using Junior < Intermediate < Senior < Expert).

###### 4. Variable domain:

$x_{w,p} \in \{0,1\} \quad \forall w \in W, p \in P$

---

##### Retrieved Information

###### Workers (with skill and on_leave status):

| worker_id | skill        | on_leave |
|-----------|-------------|----------|
| W00       | Intermediate| 0        |
| W18       | Intermediate| 0        |
| W02       | Junior      | 0        |
| W17       | Junior      | 0        |
| W10       | Expert      | 0        |
| W05       | Senior      | 0        |
| W11       | Expert      | 0        |
| W14       | Expert      | 0        |
| W01       | Intermediate| 0        |
| W24       | Expert      | 0        |
| W16       | Senior      | 0        |
| W06       | Junior      | 1        |  ← Excluded (on_leave=1)

Eligible workers: W00, W18, W02, W17, W10, W05, W11, W14, W01, W24, W16

###### Projects (with required_skill):

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

###### Skill hierarchy (for eligibility):

Junior < Intermediate < Senior < Expert

###### Cost Matrix (USD cents, blank = forbidden):

Below, only eligible assignments are shown (i.e., worker not on leave, cost cell not blank, and skill(w) ≥ required_skill(p)). For each worker, only projects for which they are eligible and have a cost entry are included.

```json
{
  "cost": {
    "W00": {"P01": 1116, "P02": 414, "P04": 554, "P05": 113, "P08": 853, "P09": 1142},
    "W01": {"P01": 170, "P03": 919, "P04": 660, "P05": 1314, "P06": 668, "P07": 4, "P08": 462, "P09": 614},
    "W02": {"P00": 330, "P01": 193, "P03": 4, "P04": 431, "P06": 1130, "P07": 15, "P08": 644, "P09": 9},
    "W05": {"P00": 325, "P01": 772, "P02": 1042, "P04": 1394, "P06": 374, "P07": 2, "P08": 1140, "P09": 127},
    "W10": {"P00": 1079, "P01": 1128, "P02": 758, "P05": 108, "P07": 423, "P08": 744, "P09": 347},
    "W11": {"P00": 1143, "P01": 841, "P02": 920, "P03": 122, "P04": 634, "P06": 1250, "P07": 836, "P08": 180, "P09": 1105},
    "W14": {"P01": 687, "P02": 126, "P03": 547, "P04": 125, "P05": 1358, "P06": 1391, "P07": 814, "P08": 1362, "P09": 962},
    "W16": {"P00": 265, "P01": 1114, "P03": 1209, "P04": 231, "P05": 221, "P08": 135},
    "W17": {"P00": 538, "P02": 1359, "P03": 20, "P04": 1366, "P05": 702, "P06": 122, "P07": 13, "P08": 358},
    "W18": {"P00": 238, "P02": 1085, "P03": 529, "P04": 582, "P05": 889, "P06": 139, "P07": 18, "P08": 630, "P09": 538},
    "W24": {"P00": 784, "P01": 1178, "P02": 1171, "P04": 103, "P05": 137, "P06": 832, "P07": 1279, "P08": 893, "P09": 351}
  },
  "workers": [
    {"worker_id": "W00", "skill": "Intermediate"},
    {"worker_id": "W01", "skill": "Intermediate"},
    {"worker_id": "W02", "skill": "Junior"},
    {"worker_id": "W05", "skill": "Senior"},
    {"worker_id": "W10", "skill": "Expert"},
    {"worker_id": "W11", "skill": "Expert"},
    {"worker_id": "W14", "skill": "Expert"},
    {"worker_id": "W16", "skill": "Senior"},
    {"worker_id": "W17", "skill": "Junior"},
    {"worker_id": "W18", "skill": "Intermediate"},
    {"worker_id": "W24", "skill": "Expert"}
  ],
  "projects": [
    {"project_id": "P00", "required_skill": "Junior"},
    {"project_id": "P01", "required_skill": "Junior"},
    {"project_id": "P02", "required_skill": "Junior"},
    {"project_id": "P03", "required_skill": "Intermediate"},
    {"project_id": "P04", "required_skill": "Junior"},
    {"project_id": "P05", "required_skill": "Junior"},
    {"project_id": "P06", "required_skill": "Junior"},
    {"project_id": "P07", "required_skill": "Expert"},
    {"project_id": "P08", "required_skill": "Junior"},
    {"project_id": "P09", "required_skill": "Intermediate"}
  ]
}
```

- Only $x_{w,p}$ for which $c_{w,p}$ is defined and $skill(w) \geq required\_skill(p)$ are allowed to be nonzero; all others are fixed to $0$.

##### Variable domains and eligibility are enforced as described above. The minimum total cost (in USD cents) is the optimal value of the objective function.