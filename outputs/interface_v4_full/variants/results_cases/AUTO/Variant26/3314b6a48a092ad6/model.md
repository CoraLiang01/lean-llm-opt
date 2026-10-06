##### Sets and Indices

- Let $W$ be the set of eligible workers (on_leave = 0):  
  $W = \{\text{W00}, \text{W04}, \text{W06}, \text{W10}, \text{W11}, \text{W14}, \text{W15}, \text{W19}, \text{W20}\}$

- Let $P$ be the set of projects:  
  $P = \{\text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}, \text{P06}, \text{P07}\}$

- Let $O \subseteq W \times P$ be the set of permitted offers (from the offer list), after filtering for on_leave = 0 and skill eligibility.

##### Parameters

- $c_{w,p}$: Cost in USD cents for worker $w$ to do project $p$, for each $(w,p) \in O$.

##### Variables

- $x_{w,p} \in \{0,1\}$: $1$ if worker $w$ is assigned to project $p$, $0$ otherwise, for each $(w,p) \in O$.

##### Objective Function

$\min \sum_{(w,p) \in O} c_{w,p} x_{w,p}$

##### Constraints

1. **Each project is assigned to exactly one worker:**

$\sum_{\substack{w: (w,p) \in O}} x_{w,p} = 1 \quad \forall p \in P$

2. **Each worker is assigned to at most one project:**

$\sum_{\substack{p: (w,p) \in O}} x_{w,p} \leq 1 \quad \forall w \in W$

3. **Assignment only allowed for eligible offers:**

$x_{w,p} = 0$ for all $(w,p) \notin O$

4. **Variable domain:**

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in O$

##### Retrieved Information

```json
{
  "workers": {
    "W00": {"skill": "Junior", "on_leave": 0},
    "W04": {"skill": "Junior", "on_leave": 0},
    "W06": {"skill": "Expert", "on_leave": 0},
    "W10": {"skill": "Expert", "on_leave": 0},
    "W11": {"skill": "Senior", "on_leave": 0},
    "W14": {"skill": "Intermediate", "on_leave": 0},
    "W15": {"skill": "Expert", "on_leave": 0},
    "W19": {"skill": "Junior", "on_leave": 0},
    "W20": {"skill": "Intermediate", "on_leave": 0}
  },
  "projects": {
    "P00": {"required_skill": "Intermediate"},
    "P01": {"required_skill": "Intermediate"},
    "P02": {"required_skill": "Senior"},
    "P03": {"required_skill": "Junior"},
    "P04": {"required_skill": "Junior"},
    "P05": {"required_skill": "Junior"},
    "P06": {"required_skill": "Junior"},
    "P07": {"required_skill": "Junior"}
  },
  "offers": [
    {"worker_id": "W10", "project_id": "P00", "cost_cents": 147, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W11", "project_id": "P00", "cost_cents": 235, "worker_skill": "Senior", "required_skill": "Intermediate"},
    {"worker_id": "W20", "project_id": "P00", "cost_cents": 675, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W06", "project_id": "P00", "cost_cents": 1217, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W15", "project_id": "P00", "cost_cents": 722, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W10", "project_id": "P01", "cost_cents": 114, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W20", "project_id": "P01", "cost_cents": 1055, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W06", "project_id": "P01", "cost_cents": 425, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W14", "project_id": "P01", "cost_cents": 1361, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W11", "project_id": "P02", "cost_cents": 1034, "worker_skill": "Senior", "required_skill": "Senior"},
    {"worker_id": "W06", "project_id": "P02", "cost_cents": 1320, "worker_skill": "Expert", "required_skill": "Senior"},
    {"worker_id": "W15", "project_id": "P02", "cost_cents": 577, "worker_skill": "Expert", "required_skill": "Senior"},
    {"worker_id": "W10", "project_id": "P03", "cost_cents": 998, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P03", "cost_cents": 109, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P03", "cost_cents": 782, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P03", "cost_cents": 703, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P03", "cost_cents": 543, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P03", "cost_cents": 1097, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P03", "cost_cents": 1062, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P03", "cost_cents": 379, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W10", "project_id": "P04", "cost_cents": 1270, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P04", "cost_cents": 1306, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P04", "cost_cents": 556, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P04", "cost_cents": 430, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P04", "cost_cents": 205, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P04", "cost_cents": 822, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P04", "cost_cents": 1314, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P04", "cost_cents": 239, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P05", "cost_cents": 626, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P05", "cost_cents": 1018, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P05", "cost_cents": 1385, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P05", "cost_cents": 1149, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P05", "cost_cents": 221, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P05", "cost_cents": 528, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P05", "cost_cents": 460, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P06", "cost_cents": 466, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P06", "cost_cents": 260, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P06", "cost_cents": 893, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P06", "cost_cents": 533, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P06", "cost_cents": 496, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P06", "cost_cents": 835, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P06", "cost_cents": 130, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W10", "project_id": "P07", "cost_cents": 953, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P07", "cost_cents": 1365, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P07", "cost_cents": 1136, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P07", "cost_cents": 1107, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P07", "cost_cents": 129, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P07", "cost_cents": 732, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P07", "cost_cents": 908, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P07", "cost_cents": 918, "worker_skill": "Expert", "required_skill": "Junior"}
  ]
}
```

##### Notes

- Only the above offers are permitted; $x_{w,p}$ is defined only for $(w,p) \in O$.
- All costs $c_{w,p}$ are in USD cents.
- Skill eligibility and on_leave filtering have already been applied in the above offer list.

---

**Summary:**  
Minimize the total assignment cost in USD cents, assigning each project to exactly one eligible worker, with each worker assigned to at most one project, using only the listed offers and respecting skill and leave constraints. All parameters and sets are as above.