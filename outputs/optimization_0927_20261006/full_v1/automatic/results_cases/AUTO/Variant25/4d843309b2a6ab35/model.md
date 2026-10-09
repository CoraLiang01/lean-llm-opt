##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{w,p}$ is the assignment cost in USD cents (forbidden assignments have no variable and do not appear in the sum).

##### Constraints

###### 1. Each project is assigned to exactly one eligible worker:

$\sum_{w \in W_p} x_{w,p} = 1 \quad \forall p \in P$

where $W_p$ is the set of eligible workers for project $p$ (not on leave, skill $\geq$ required, and $c_{w,p}$ not blank).

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P_w} x_{w,p} \leq 1 \quad \forall w \in W$

where $P_w$ is the set of projects worker $w$ is eligible for.

###### 3. Assignment domain:

$x_{w,p} \in \{0,1\} \quad \forall w,p$ (only for eligible assignments).

##### Retrieved Information

```json
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
  "cost_matrix": {
    "W12": {"P00": "102", "P01": "353", "P02": "651", "P03": "102", "P04": "",    "P05": "7"},
    "W06": {"P00": "",   "P01": "822", "P02": "",   "P03": "223", "P04": "1155", "P05": "1055"},
    "W04": {"P00": "642", "P01": "",   "P02": "1130", "P03": "133", "P04": "199", "P05": "311"},
    "W11": {"P00": "1091", "P01": "324", "P02": "379", "P03": "272", "P04": "",   "P05": ""},
    "W10": {"P00": "1176", "P01": "1111", "P02": "1380", "P03": "542", "P04": "158", "P05": "922"},
    "W02": {"P00": "",   "P01": "1397", "P02": "953", "P03": "714", "P04": "205", "P05": ""},
    "W00": {"P00": "1063", "P01": "219", "P02": "",   "P03": "1329", "P04": "",   "P05": "436"}
  },
  "skill_order": ["Junior", "Intermediate", "Senior", "Expert"]
}
```

##### Eligibility (based on on_leave and skill):

- Exclude W01 (on_leave=1).
- For each assignment $(w,p)$, allow only if:
    - $c_{w,p}$ is not blank,
    - worker's skill $\geq$ required_skill for project.

##### Explicit Cost Matrix for Eligible Assignments (USD cents):

|         | P00   | P01   | P02   | P03   | P04   | P05   |
|---------|-------|-------|-------|-------|-------|-------|
| W12     | 102   | 353   | 651   | 102   |       |       |
| W06     |       | 822   |       | 223   | 1155  | 1055  |
| W04     | 642   |       | 1130  | 133   | 199   | 311   |
| W11     | 1091  | 324   | 379   | 272   |       |       |
| W10     | 1176  | 1111  | 1380  | 542   | 158   | 922   |
| W02     |       | 1397  | 953   | 714   | 205   |       |
| W00     | 1063  | 219   |       | 1329  |       | 436   |

- For P05 (required_skill=Intermediate): only W06, W04, W10, W00 are eligible (W12, W11 are Junior; W02 is Intermediate but cost is blank).

##### Variable Definitions

Let $x_{w,p}$ be defined only for eligible $(w,p)$ pairs as above.

##### Model Summary

- Minimize total cost of assignments.
- Each project assigned to exactly one eligible worker.
- Each worker assigned to at most one project.
- Only eligible assignments allowed (as per cost matrix, on_leave, and skill).

##### Minimum Cost (in USD cents):

Let $z^*$ denote the minimum value of the objective function as defined above. The model will yield $z^*$ in USD cents.