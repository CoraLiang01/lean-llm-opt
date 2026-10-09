##### Objective Function:

$\quad \min \sum_{(w,p) \in \mathcal{A}} c_{w,p} \, x_{w,p}$

where:
- $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise,
- $\mathcal{A}$ is the set of all allowed (worker, project) assignments (offers) that satisfy:
  - worker is not on leave ($\text{on\_leave}=0$),
  - worker's skill $\geq$ required skill for the project,
  - the offer is listed in the data,
- $c_{w,p}$ is the cost in USD cents for assigning worker $w$ to project $p$.

##### Constraints

###### 1. Each project is assigned exactly one worker:

$\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

where $\mathcal{P}$ is the set of all project IDs in the data.

###### 2. Each worker is assigned to at most one project:

$\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

where $\mathcal{W}$ is the set of all worker IDs in the data.

###### 3. Assignment only allowed for listed offers, not on leave, and skill-qualified:

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}$

###### 4. No assignment for unlisted, on-leave, or under-qualified offers:

$x_{w,p} = 0$ for all $(w,p)$ not in $\mathcal{A}$

##### Retrieved Information

- **Skill hierarchy:** Junior < Intermediate < Senior < Expert
- **Allowed assignments $\mathcal{A}$:** (worker_id, project_id) pairs from the data with on_leave=0 and worker_skill $\geq$ required_skill (according to the hierarchy above).
- **Cost matrix $c_{w,p}$:** as listed below for all $(w,p) \in \mathcal{A}$.

```json
{
  "offers": [
    {"worker_id": "W10", "project_id": "P00", "cost_cents": 147, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W10", "project_id": "P01", "cost_cents": 114, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W10", "project_id": "P03", "cost_cents": 998, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W10", "project_id": "P04", "cost_cents": 1270, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W10", "project_id": "P07", "cost_cents": 953, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P00", "cost_cents": 235, "worker_skill": "Senior", "required_skill": "Intermediate"},
    {"worker_id": "W11", "project_id": "P02", "cost_cents": 1034, "worker_skill": "Senior", "required_skill": "Senior"},
    {"worker_id": "W11", "project_id": "P03", "cost_cents": 782, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P04", "cost_cents": 556, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P05", "cost_cents": 1018, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W11", "project_id": "P07", "cost_cents": 1136, "worker_skill": "Senior", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P00", "cost_cents": 675, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W20", "project_id": "P01", "cost_cents": 1055, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W20", "project_id": "P02", "cost_cents": 8, "worker_skill": "Intermediate", "required_skill": "Senior"},
    {"worker_id": "W20", "project_id": "P03", "cost_cents": 703, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P06", "cost_cents": 893, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P07", "cost_cents": 129, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P01", "cost_cents": 1361, "worker_skill": "Intermediate", "required_skill": "Intermediate"},
    {"worker_id": "W14", "project_id": "P03", "cost_cents": 379, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P04", "cost_cents": 239, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P05", "cost_cents": 460, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W14", "project_id": "P06", "cost_cents": 130, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P00", "cost_cents": 722, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W15", "project_id": "P02", "cost_cents": 577, "worker_skill": "Expert", "required_skill": "Senior"},
    {"worker_id": "W15", "project_id": "P03", "cost_cents": 1062, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P04", "cost_cents": 1314, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P05", "cost_cents": 528, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P06", "cost_cents": 835, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W15", "project_id": "P07", "cost_cents": 918, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P00", "cost_cents": 1217, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W06", "project_id": "P01", "cost_cents": 425, "worker_skill": "Expert", "required_skill": "Intermediate"},
    {"worker_id": "W06", "project_id": "P02", "cost_cents": 1320, "worker_skill": "Expert", "required_skill": "Senior"},
    {"worker_id": "W06", "project_id": "P03", "cost_cents": 1097, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P04", "cost_cents": 822, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P05", "cost_cents": 221, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P06", "cost_cents": 496, "worker_skill": "Expert", "required_skill": "Junior"},
    {"worker_id": "W06", "project_id": "P07", "cost_cents": 908, "worker_skill": "Expert", "required_skill": "Junior"}
  ],
  "projects": ["P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07"],
  "workers": ["W10", "W11", "W20", "W14", "W15", "W06"]
}
```

- Only the above (worker, project) pairs are allowed for assignment.
- All other offers (including those with on_leave=1 or insufficient skill) are excluded.

##### Variable Domains

$x_{w,p} \in \{0,1\}$ for all $(w,p) \in \mathcal{A}$

##### Summary

- Assign each project exactly one worker.
- Each worker may be assigned to at most one project.
- Only listed, not-on-leave, skill-qualified offers are allowed.
- Minimize total cost in USD cents.