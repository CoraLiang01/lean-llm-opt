##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents.

##### Constraints

###### 1. Project Assignment (each project assigned to exactly one eligible worker):

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Worker Assignment (each worker assigned to at most one project):

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Eligibility Mask (assignment allowed only if all conditions are met):

For each $(w,p)$ pair, $x_{w,p}$ is allowed only if:
- Worker $w$ is not on leave ($on\_leave = 0$)
- Worker $w$'s skill $\geq$ project $p$'s required_skill (Junior < Intermediate < Senior < Expert)
- The cost $c_{w,p}$ is not blank

For all ineligible $(w,p)$ pairs, add:

$x_{w,p} = 0$

###### 4. Variable Domain:

$x_{w,p} \in \{0,1\} \quad \forall w \in W,\, p \in P$

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

(Excluded: W01, on_leave=1)

###### Projects:

| project_id | required_skill |
|------------|---------------|
| P00        | Junior        |
| P01        | Junior        |
| P02        | Junior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Intermediate  |

###### Skill Levels (for comparison):

Junior < Intermediate < Senior < Expert

###### Cost Matrix (only eligible, nonblank pairs):

| worker_id | skill        | P00  | P01  | P02  | P03  | P04  | P05  |
|-----------|-------------|------|------|------|------|------|------|
| W12       | Junior      | 102  | 353  | 651  | 102  |      | 7    |
| W06       | Expert      |      | 822  |      | 223  | 1155 | 1055 |
| W04       | Intermediate| 642  |      | 1130 | 133  | 199  | 311  |
| W11       | Junior      | 1091 | 324  | 379  | 272  |      |      |
| W10       | Intermediate| 1176 | 1111 | 1380 | 542  | 158  | 922  |
| W02       | Intermediate|      | 1397 | 953  | 714  | 205  |      |
| W00       | Expert      | 1063 | 219  |      | 1329 |      | 436  |

Eligibility mask (1 = eligible, 0 = ineligible):

- For each $(w,p)$, eligible if:
    - $on\_leave = 0$
    - $skill(w) \geq required\_skill(p)$
    - $c_{w,p}$ is not blank

Eligible assignment pairs (with costs):

- W12 (Junior): P00 (102), P01 (353), P02 (651), P03 (102), P05 (7)
- W06 (Expert): P01 (822), P03 (223), P04 (1155), P05 (1055)
- W04 (Intermediate): P00 (642), P02 (1130), P03 (133), P04 (199), P05 (311)
- W11 (Junior): P00 (1091), P01 (324), P02 (379), P03 (272)
- W10 (Intermediate): P00 (1176), P01 (1111), P02 (1380), P03 (542), P04 (158), P05 (922)
- W02 (Intermediate): P01 (1397), P02 (953), P03 (714), P04 (205)
- W00 (Expert): P00 (1063), P01 (219), P03 (1329), P05 (436)

Ineligible pairs (must set $x_{w,p}=0$):

- Any blank cell in the cost matrix
- Any pair where worker's skill < required_skill (e.g., W12 and W11 cannot do P05, since both are Junior and P05 requires Intermediate)
- Any worker on leave (W01, all projects)

##### Sets

$W = \{$W12, W06, W04, W11, W10, W02, W00$\}$

$P = \{$P00, P01, P02, P03, P04, P05$\}$

##### Parameters

$c_{w,p}$ as above, only for eligible pairs.

##### Decision Variables

$x_{w,p} \in \{0,1\}$ for eligible $(w,p)$ pairs; $x_{w,p}=0$ for all other pairs.

---

##### Full Mathematical Model

$\min \sum_{(w,p) \in E} c_{w,p} \, x_{w,p}$

subject to

$\sum_{w: (w,p) \in E} x_{w,p} = 1 \quad \forall p \in P$

$\sum_{p: (w,p) \in E} x_{w,p} \leq 1 \quad \forall w \in W$

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in E$

$x_{w,p} = 0 \quad \forall (w,p) \notin E$

where $E$ is the set of eligible $(w,p)$ pairs as defined above.

---

##### Retrieved Information (Structured)

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