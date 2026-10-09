##### Sets and Indices

Let $W$ be the set of workers and $P$ the set of projects. Let $O$ be the set of feasible offers, i.e., tuples $(w,p)$ where worker $w$ is not on leave, the offer is listed, and $w$'s skill is at least the required skill for $p$.

Let $x_{w,p} \in \{0,1\}$ be a binary variable indicating whether worker $w$ is assigned to project $p$.

##### Parameters

- $c_{w,p}$: cost in USD cents for worker $w$ to do project $p$, as given in the offer list.
- $W = \{$W00, W04, W06, W10, W11, W14, W15, W19, W20$\}$
- $P = \{$P00, P01, P02, P03, P04, P05, P06, P07$\}$
- $O$: the set of all $(w,p)$ pairs below (see "Retrieved Information").

##### Objective Function

$\min \sum_{(w,p) \in O} c_{w,p} x_{w,p}$

##### Constraints

1. **Each project is assigned to exactly one worker:**

$\sum_{w: (w,p) \in O} x_{w,p} = 1 \quad \forall p \in P$

2. **Each worker is assigned to at most one project:**

$\sum_{p: (w,p) \in O} x_{w,p} \leq 1 \quad \forall w \in W$

3. **Assignment only allowed for listed, eligible offers:**

$x_{w,p} = 0$ for all $(w,p) \notin O$

4. **Variable domain:**

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
    {"worker_id": "W20", "project_id": "P03", "cost_cents": 703, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P06", "cost_cents": 893, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
    {"worker_id": "W20", "project_id": "P07", "cost_cents": 129, "on_leave": 0, "worker_skill": "Intermediate", "required_skill": "Junior"},
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
  ]
}
```

- Only the above offers are permitted.
- All workers listed above have on_leave=0 and are eligible for their respective offers.
- Each assignment must respect the skill hierarchy: Junior < Intermediate < Senior < Expert.

##### Variable Domains

$x_{w,p} \in \{0,1\}$ for all $(w,p) \in O$.

##### Summary

Minimize the total cost in USD cents of assigning workers to projects, using only the listed, eligible offers, such that each project is covered by exactly one worker, and each worker is assigned to at most one project.