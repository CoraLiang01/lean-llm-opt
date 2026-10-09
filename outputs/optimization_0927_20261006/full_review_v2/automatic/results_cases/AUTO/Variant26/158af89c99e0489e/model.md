##### Sets and Indices

Let $W$ be the set of eligible workers (not on leave), $P$ the set of projects. Let $O$ be the set of all listed offers, each as a tuple $(w,p)$ with $w \in W$, $p \in P$.

##### Parameters

- $c_{w,p}$: Assignment cost in USD cents for worker $w$ to project $p$, as given in the offer list.
- $S_w$: Skill level of worker $w$ (Junior $=1$, Intermediate $=2$, Senior $=3$, Expert $=4$).
- $R_p$: Required skill level for project $p$ (Junior $=1$, Intermediate $=2$, Senior $=3$, Expert $=4$).

##### Variables

- $x_{w,p} \in \{0,1\}$: $1$ if worker $w$ is assigned to project $p$, $0$ otherwise. Only defined for $(w,p) \in O$.

##### Objective Function

$\min \sum_{(w,p) \in O} c_{w,p} \, x_{w,p}$

##### Constraints

1. **Project Coverage:** Each project is assigned to exactly one worker (from eligible offers):

$\sum_{\substack{w: (w,p) \in O \\ S_w \geq R_p}} x_{w,p} = 1 \quad \forall p \in P$

2. **Worker Assignment:** Each worker is assigned to at most one project:

$\sum_{\substack{p: (w,p) \in O \\ S_w \geq R_p}} x_{w,p} \leq 1 \quad \forall w \in W$

3. **Skill Feasibility:** Only allow assignments where the worker's skill meets or exceeds the required skill:

$x_{w,p} = 0 \quad \text{if } S_w < R_p \quad \forall (w,p) \in O$

4. **Offer List Restriction:** Only listed offers are permitted; $x_{w,p}$ is defined only for $(w,p) \in O$.

5. **On Leave Exclusion:** Only workers with $on\_leave = 0$ are included in $W$ and $O$.

6. **Binary Variables:**

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in O$

##### Retrieved Information

```json
{
  "offers": [
    {"worker_id": "W10", "project_id": "P00", "cost_cents": 147, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W10", "project_id": "P01", "cost_cents": 114, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W10", "project_id": "P03", "cost_cents": 998, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W10", "project_id": "P04", "cost_cents": 1270, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W10", "project_id": "P07", "cost_cents": 953, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P00", "cost_cents": 5, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Intermediate"},
    {"worker_id": "W19", "project_id": "P02", "cost_cents": 9, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Senior"},
    {"worker_id": "W19", "project_id": "P03", "cost_cents": 109, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P04", "cost_cents": 1306, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P05", "cost_cents": 626, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P06", "cost_cents": 466, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W19", "project_id": "P07", "cost_cents": 1365, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P00", "cost_cents": 235, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Intermediate"},
    {"worker_id": "W11", "project_id": "P02", "cost_cents": 1034, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Senior"},
    {"worker_id": "W11", "project_id": "P03", "cost_cents": 782, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P04", "cost_cents": 556, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P05", "cost_cents": 1018, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P07", "cost_cents": 1136, "on_leave": 0, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P01", "cost_cents": 17, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Intermediate"},
    {"worker_id": "W00", "project_id": "P04", "cost_cents": 430, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P05", "cost_cents": 1385, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P06", "cost_cents": 260, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W00", "project_id": "P07", "cost_cents": 1107, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P00", "cost_cents": 675, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W20", "project_id": "P01", "cost_cents": 1055, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W20", "project_id": "P02", "cost_cents": 8, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Senior"},
    {"worker_id": "W20", "project_id": "P03", "cost_cents": 703, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P06", "cost_cents": 893, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P07", "cost_cents": 129, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P00", "cost_cents": 4, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Intermediate"},
    {"worker_id": "W04", "project_id": "P01", "cost_cents": 8, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Intermediate"},
    {"worker_id": "W04", "project_id": "P02", "cost_cents": 16, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Senior"},
    {"worker_id": "W04", "project_id": "P03", "cost_cents": 543, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P04", "cost_cents": 205, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P05", "cost_cents": 1149, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P06", "cost_cents": 533, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W04", "project_id": "P07", "cost_cents": 732, "on_leave": 0, "worker_skill": "Junior", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P00", "cost_cents": 1217, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W06", "project_id": "P01", "cost_cents": 425, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W06", "project_id": "P02", "cost_cents": 1320, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Senior"},
    {"worker_id": "W06", "project_id": "P03", "cost_cents": 1097, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P04", "cost_cents": 822, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P05", "cost_cents": 221, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P06", "cost_cents": 496, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P07", "cost_cents": 908, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P01", "cost_cents": 1361, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W14", "project_id": "P03", "cost_cents": 379, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P04", "cost_cents": 239, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P05", "cost_cents": 460, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P06", "cost_cents": 130, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P00", "cost_cents": 722, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W15", "project_id": "P02", "cost_cents": 577, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Senior"},
    {"worker_id": "W15", "project_id": "P03", "cost_cents": 1062, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P04", "cost_cents": 1314, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P05", "cost_cents": 528, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P06", "cost_cents": 835, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P07", "cost_cents": 918, "on_leave": 0, "worker_skill": "Expert", "required_skill": "Junior"}
  ],
  "skill_levels": {
    "Junior": 1,
    "Intermediate": 2,
    "Senior": 3,
    "Expert": 4
  },
  "projects": [
    "P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07"
  ],
  "workers": [
    "W00", "W04", "W06", "W10", "W11", "W14", "W15", "W19", "W20"
  ]
}
```

##### Notes

- Only offers with $on\_leave = 0$ are included.
- Only listed offers are permitted; if a $(w,p)$ pair is not in the offer list, $x_{w,p}$ is not defined.
- Skill levels are mapped as: Junior $=1$, Intermediate $=2$, Senior $=3$, Expert $=4$.
- The minimum total cost is reported in USD cents.