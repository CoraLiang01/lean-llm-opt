##### Objective Function:

$\quad \min \sum_{(w,p) \in \mathcal{A}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

##### Sets and Indices

- $\mathcal{W}$: Set of eligible workers (from file_0_view_0, where on_leave = 0)
- $\mathcal{P}$: Set of projects (from file_1_view_0)
- $\mathcal{A}$: Set of allowed (worker, project) assignments (from file_2_view_0, i.e., all listed offers with eligible workers)
- $\text{skill}_w$: Skill level of worker $w$ (from file_0_view_0)
- $\text{required\_skill}_p$: Required skill for project $p$ (from file_1_view_0)

##### Decision Variables

- $x_{w,p} \in \{0,1\}$ for $(w,p) \in \mathcal{A}$  
 $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise.

##### Constraints

1. **Each project assigned to exactly one worker:**

   $\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

2. **Each worker assigned to at most one project:**

   $\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

3. **Skill requirement:**

   $x_{w,p} = 0$ unless $\text{skill}_w \geq \text{required\_skill}_p$ (using the order: Junior < Intermediate < Senior < Expert), for all $(w,p) \in \mathcal{A}$

4. **Eligibility and offer restriction:**

   $x_{w,p}$ is only defined for $(w,p) \in \mathcal{A}$ (i.e., only listed offers for workers not on leave).

##### Variable Domains

- $x_{w,p} \in \{0,1\}$ for all $(w,p) \in \mathcal{A}$

---

##### Retrieved Information

```json
{
  "workers": {
    "table_id": "file_0_view_0",
    "columns": ["worker_id", "skill", "on_leave"],
    "eligible_workers": [
      {"worker_id": "W00", "skill": "Junior", "on_leave": "0"},
      {"worker_id": "W19", "skill": "Junior", "on_leave": "0"},
      {"worker_id": "W11", "skill": "Senior", "on_leave": "0"},
      {"worker_id": "W04", "skill": "Junior", "on_leave": "0"},
      {"worker_id": "W15", "skill": "Expert", "on_leave": "0"},
      {"worker_id": "W10", "skill": "Expert", "on_leave": "0"},
      {"worker_id": "W06", "skill": "Expert", "on_leave": "0"},
      {"worker_id": "W20", "skill": "Intermediate", "on_leave": "0"},
      {"worker_id": "W14", "skill": "Intermediate", "on_leave": "0"}
    ]
  },
  "projects": {
    "table_id": "file_1_view_0",
    "columns": ["project_id", "required_skill"]
  },
  "offers": {
    "table_id": "file_2_view_0",
    "columns": ["worker_id", "project_id", "cost_cents"]
  }
}
```

- $\mathcal{W}$: All worker_id in eligible_workers (on_leave = 0) from file_0_view_0
- $\mathcal{P}$: All project_id from file_1_view_0
- $\mathcal{A}$: All (worker_id, project_id) pairs listed in file_2_view_0 where worker_id is in $\mathcal{W}$
- $\text{cost\_cents}_{w,p}$: cost_cents from file_2_view_0 for each $(w,p) \in \mathcal{A}$
- $\text{skill}_w$: skill from file_0_view_0 for each $w \in \mathcal{W}$
- $\text{required\_skill}_p$: required_skill from file_1_view_0 for each $p \in \mathcal{P}$

---

**All indices, parameters, and constraints are mapped directly to the provided CSV data as described above.**