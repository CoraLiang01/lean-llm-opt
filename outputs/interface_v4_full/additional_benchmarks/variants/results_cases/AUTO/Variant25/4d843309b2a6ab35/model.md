##### Objective Function:

$\quad \min \sum_{i \in \mathcal{W}} \sum_{j \in \mathcal{P}} c_{ij} x_{ij}$

where $x_{ij} = 1$ if worker $i$ is assigned to project $j$, $0$ otherwise, and $c_{ij}$ is the assignment cost in USD cents (forbidden assignments have $x_{ij}$ forced to $0$).

##### Constraints

###### 1. Each project is assigned to exactly one eligible worker:

$\sum_{i \in \mathcal{W}_j} x_{ij} = 1 \quad \forall j \in \mathcal{P}$

where $\mathcal{W}_j$ is the set of eligible workers for project $j$ (not on leave, skill $\geq$ required_skill, and $c_{ij}$ not blank).

###### 2. Each worker is assigned to at most one project:

$\sum_{j \in \mathcal{P}_i} x_{ij} \leq 1 \quad \forall i \in \mathcal{W}$

where $\mathcal{P}_i$ is the set of projects worker $i$ is eligible for.

###### 3. Assignment variable domain and forbidden assignments:

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{W}, j \in \mathcal{P}$

If $c_{ij}$ is blank, or worker $i$ is on leave, or worker $i$'s skill $<$ required_skill for project $j$, then $x_{ij} = 0$.

##### Retrieved Information

{
  "workers": [
    {"id": "W12", "skill": "Junior", "on_leave": 0},
    {"id": "W06", "skill": "Expert", "on_leave": 0},
    {"id": "W04", "skill": "Intermediate", "on_leave": 0},
    {"id": "W11", "skill": "Junior", "on_leave": 0},
    {"id": "W10", "skill": "Intermediate", "on_leave": 0},
    {"id": "W02", "skill": "Intermediate", "on_leave": 0},
    {"id": "W00", "skill": "Expert", "on_leave": 0}
  ],
  "projects": [
    {"id": "P00", "required_skill": "Junior"},
    {"id": "P01", "required_skill": "Junior"},
    {"id": "P02", "required_skill": "Junior"},
    {"id": "P03", "required_skill": "Junior"},
    {"id": "P04", "required_skill": "Junior"},
    {"id": "P05", "required_skill": "Intermediate"}
  ],
  "cost": {
    "W12": {"P00": 102, "P01": 353, "P02": 651, "P03": 102, "P04": null, "P05": 7},
    "W06": {"P00": null, "P01": 822, "P02": null, "P03": 223, "P04": 1155, "P05": 1055},
    "W04": {"P00": 642, "P01": null, "P02": 1130, "P03": 133, "P04": 199, "P05": 311},
    "W11": {"P00": 1091, "P01": 324, "P02": 379, "P03": 272, "P04": null, "P05": null},
    "W10": {"P00": 1176, "P01": 1111, "P02": 1380, "P03": 542, "P04": 158, "P05": 922},
    "W02": {"P00": null, "P01": 1397, "P02": 953, "P03": 714, "P04": 205, "P05": null},
    "W00": {"P00": 1063, "P01": 219, "P02": null, "P03": 1329, "P04": null, "P05": 436}
  },
  "skill_order": ["Junior", "Intermediate", "Senior", "Expert"]
}

##### Eligible Assignments

- For each $(i, j)$, $x_{ij}$ is allowed only if:
    - Worker $i$ is not on leave,
    - Worker $i$'s skill $\geq$ required_skill for project $j$ (using the order: Junior < Intermediate < Senior < Expert),
    - $c_{ij}$ is not blank.

##### Variable Domain

$x_{ij} \in \{0,1\}$ for all eligible $(i, j)$; $x_{ij} = 0$ for forbidden assignments.

##### Objective

Minimize the total assignment cost in USD cents:

$\min \sum_{i \in \mathcal{W}} \sum_{j \in \mathcal{P}} c_{ij} x_{ij}$

where $c_{ij}$ and eligibility are as above.