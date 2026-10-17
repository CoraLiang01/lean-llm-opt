##### Objective Function:

$\quad \min \sum_{(w,p) \in E} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $E$ is the set of eligible (worker, project) pairs as defined below.

##### Eligibility Mask (Pair Set $E$):

A pair $(w,p)$ is eligible and included in $E$ if:
- The offer $(w,p)$ is listed in the data (only listed offers are permitted).
- Worker $w$ is not on leave ($\text{on\_leave}_w = 0$).
- Worker $w$'s skill is at least as high as the required skill for project $p$ (with Junior < Intermediate < Senior < Expert).

##### Constraints

###### 1. Project Coverage:

$\sum_{w: (w,p) \in E} x_{w,p} = 1 \quad \forall p \in P$

(Each project is assigned exactly one eligible worker.)

###### 2. Worker Assignment:

$\sum_{p: (w,p) \in E} x_{w,p} \leq 1 \quad \forall w \in W$

(Each worker is assigned to at most one project.)

###### 3. Variable Domain:

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in E$

##### Retrieved Information

###### Workers (with skill and leave status):

```json
{
  "W00": {"skill": "Junior", "on_leave": 0},
  "W01": {"skill": "Senior", "on_leave": 1},
  "W04": {"skill": "Junior", "on_leave": 0},
  "W06": {"skill": "Expert", "on_leave": 0},
  "W10": {"skill": "Expert", "on_leave": 0},
  "W11": {"skill": "Senior", "on_leave": 0},
  "W14": {"skill": "Intermediate", "on_leave": 0},
  "W15": {"skill": "Expert", "on_leave": 0},
  "W19": {"skill": "Junior", "on_leave": 0},
  "W20": {"skill": "Intermediate", "on_leave": 0}
}
```

###### Projects (with required skill):

```json
{
  "P00": "Intermediate",
  "P01": "Intermediate",
  "P02": "Senior",
  "P03": "Junior",
  "P04": "Junior",
  "P05": "Junior",
  "P06": "Junior",
  "P07": "Junior"
}
```

###### Skill Order (for comparison):

Junior < Intermediate < Senior < Expert

###### Listed Offers (before eligibility filtering):

Each offer is a tuple (worker_id, project_id, cost_cents). For brevity, only eligible offers are listed below.

###### Eligible Offers (after applying on_leave and skill rules):

- Only workers with on_leave = 0 are eligible.
- For each offer, worker's skill must be $\geq$ required_skill for the project.

Below is the full eligible offer list (with cost_cents):

```json
[
  {"worker_id": "W00", "project_id": "P01", "cost_cents": 17},
  {"worker_id": "W00", "project_id": "P04", "cost_cents": 430},
  {"worker_id": "W00", "project_id": "P05", "cost_cents": 1385},
  {"worker_id": "W00", "project_id": "P06", "cost_cents": 260},
  {"worker_id": "W00", "project_id": "P07", "cost_cents": 1107},

  {"worker_id": "W04", "project_id": "P03", "cost_cents": 543},
  {"worker_id": "W04", "project_id": "P04", "cost_cents": 205},
  {"worker_id": "W04", "project_id": "P05", "cost_cents": 1149},
  {"worker_id": "W04", "project_id": "P06", "cost_cents": 533},
  {"worker_id": "W04", "project_id": "P07", "cost_cents": 732},

  {"worker_id": "W06", "project_id": "P00", "cost_cents": 1217},
  {"worker_id": "W06", "project_id": "P01", "cost_cents": 425},
  {"worker_id": "W06", "project_id": "P02", "cost_cents": 1320},
  {"worker_id": "W06", "project_id": "P03", "cost_cents": 1097},
  {"worker_id": "W06", "project_id": "P04", "cost_cents": 822},
  {"worker_id": "W06", "project_id": "P05", "cost_cents": 221},
  {"worker_id": "W06", "project_id": "P06", "cost_cents": 496},
  {"worker_id": "W06", "project_id": "P07", "cost_cents": 908},

  {"worker_id": "W10", "project_id": "P00", "cost_cents": 147},
  {"worker_id": "W10", "project_id": "P01", "cost_cents": 114},
  {"worker_id": "W10", "project_id": "P03", "cost_cents": 998},
  {"worker_id": "W10", "project_id": "P04", "cost_cents": 1270},
  {"worker_id": "W10", "project_id": "P07", "cost_cents": 953},

  {"worker_id": "W11", "project_id": "P00", "cost_cents": 235},
  {"worker_id": "W11", "project_id": "P02", "cost_cents": 1034},
  {"worker_id": "W11", "project_id": "P03", "cost_cents": 782},
  {"worker_id": "W11", "project_id": "P04", "cost_cents": 556},
  {"worker_id": "W11", "project_id": "P05", "cost_cents": 1018},
  {"worker_id": "W11", "project_id": "P07", "cost_cents": 1136},

  {"worker_id": "W14", "project_id": "P03", "cost_cents": 379},
  {"worker_id": "W14", "project_id": "P04", "cost_cents": 239},
  {"worker_id": "W14", "project_id": "P05", "cost_cents": 460},
  {"worker_id": "W14", "project_id": "P06", "cost_cents": 130},

  {"worker_id": "W15", "project_id": "P00", "cost_cents": 722},
  {"worker_id": "W15", "project_id": "P02", "cost_cents": 577},
  {"worker_id": "W15", "project_id": "P03", "cost_cents": 1062},
  {"worker_id": "W15", "project_id": "P04", "cost_cents": 1314},
  {"worker_id": "W15", "project_id": "P05", "cost_cents": 528},
  {"worker_id": "W15", "project_id": "P06", "cost_cents": 835},
  {"worker_id": "W15", "project_id": "P07", "cost_cents": 918},

  {"worker_id": "W19", "project_id": "P03", "cost_cents": 109},
  {"worker_id": "W19", "project_id": "P04", "cost_cents": 1306},
  {"worker_id": "W19", "project_id": "P05", "cost_cents": 626},
  {"worker_id": "W19", "project_id": "P06", "cost_cents": 466},
  {"worker_id": "W19", "project_id": "P07", "cost_cents": 1365},

  {"worker_id": "W20", "project_id": "P00", "cost_cents": 675},
  {"worker_id": "W20", "project_id": "P03", "cost_cents": 703},
  {"worker_id": "W20", "project_id": "P06", "cost_cents": 893},
  {"worker_id": "W20", "project_id": "P07", "cost_cents": 129}
]
```

##### Sets

- $W$ = {W00, W04, W06, W10, W11, W14, W15, W19, W20} (workers not on leave)
- $P$ = {P00, P01, P02, P03, P04, P05, P06, P07} (projects)
- $E$ = eligible (worker, project) pairs as above

##### Variables

- $x_{w,p} \in \{0,1\}$ for each $(w,p) \in E$

##### Parameters

- $\text{cost\_cents}_{w,p}$ as listed above for each $(w,p) \in E$

##### Model Summary

Minimize the total assignment cost in USD cents, assigning each project exactly one eligible worker, each worker to at most one project, using only the listed eligible offers.