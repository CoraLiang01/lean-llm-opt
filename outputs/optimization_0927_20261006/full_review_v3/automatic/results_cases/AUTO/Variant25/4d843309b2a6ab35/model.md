##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where $x_{wp} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $c_{wp}$ is the assignment cost in USD cents (forbidden assignments have $x_{wp}$ fixed to $0$).

##### Constraints

###### 1. Each project is assigned to exactly one eligible worker:

$\sum_{w \in W_p} x_{wp} = 1 \quad \forall p \in P$

where $W_p$ is the set of eligible workers for project $p$ (not on leave, skill $\geq$ required, and $c_{wp}$ not blank).

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P_w} x_{wp} \leq 1 \quad \forall w \in W$

where $P_w$ is the set of projects worker $w$ is eligible for.

###### 3. Assignment forbidden if:
- Worker is on leave,
- Worker skill $<$ required_skill for project,
- $c_{wp}$ is blank.

For such $(w,p)$, $x_{wp} = 0$.

###### 4. Variable domains:

$x_{wp} \in \{0,1\} \quad \forall w \in W,\, p \in P$

---

##### Retrieved Information

```json
{
  "workers": [
    {"worker_id": "W12", "skill": "Junior", "on_leave": "0"},
    {"worker_id": "W06", "skill": "Expert", "on_leave": "0"},
    {"worker_id": "W04", "skill": "Intermediate", "on_leave": "0"},
    {"worker_id": "W01", "skill": "Junior", "on_leave": "1"},
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
    "W11": {"P00": "1091", "P01": "324", "P02": "379", "P03": "272", "P04": "", "P05": ""},
    "W10": {"P00": "1176", "P01": "1111", "P02": "1380", "P03": "542", "P04": "158", "P05": "922"},
    "W06": {"P00": "", "P01": "822", "P02": "", "P03": "223", "P04": "1155", "P05": "1055"},
    "W01": {"P00": "5", "P01": "", "P02": "20", "P03": "", "P04": "18", "P05": "7"},
    "W02": {"P00": "", "P01": "1397", "P02": "953", "P03": "714", "P04": "205", "P05": ""},
    "W00": {"P00": "1063", "P01": "219", "P02": "", "P03": "1329", "P04": "", "P05": "436"},
    "W04": {"P00": "642", "P01": "", "P02": "1130", "P03": "133", "P04": "199", "P05": "311"},
    "W12": {"P00": "102", "P01": "353", "P02": "651", "P03": "102", "P04": "", "P05": "7"}
  }
}
```

###### Skill hierarchy (for eligibility):

- Junior = 1
- Intermediate = 2
- Senior = 3
- Expert = 4

###### Eligible workers (not on leave):

- W12 (Junior)
- W06 (Expert)
- W04 (Intermediate)
- W11 (Junior)
- W10 (Intermediate)
- W02 (Intermediate)
- W00 (Expert)

###### Projects and required_skill:

- P00: Junior
- P01: Junior
- P02: Junior
- P03: Junior
- P04: Junior
- P05: Intermediate

###### Cost matrix (USD cents, blank = forbidden):

|         | P00   | P01   | P02   | P03   | P04   | P05   |
|---------|-------|-------|-------|-------|-------|-------|
| W12     | 102   | 353   | 651   | 102   |       | 7     |
| W06     |       | 822   |       | 223   | 1155  | 1055  |
| W04     | 642   |       | 1130  | 133   | 199   | 311   |
| W11     | 1091  | 324   | 379   | 272   |       |       |
| W10     | 1176  | 1111  | 1380  | 542   | 158   | 922   |
| W02     |       | 1397  | 953   | 714   | 205   |       |
| W00     | 1063  | 219   |       | 1329  |       | 436   |

###### Assignment eligibility:

- Exclude W01 (on_leave=1).
- For each (worker, project), allow assignment only if:
    - cost is not blank,
    - worker skill $\geq$ required_skill.

###### Variable definitions:

- $x_{wp} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, for all eligible $(w,p)$ pairs.

---

This model assigns each of the six projects to one eligible manager, with each manager assigned at most one project, minimizing the total cost in USD cents. Forbidden assignments (due to leave, insufficient skill, or blank cost) are not allowed.