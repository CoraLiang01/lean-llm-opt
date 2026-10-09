##### Sets and Parameters

Let  
- $W$ = set of workers with on\_leave = 0 and at least one listed offer  
- $P$ = set of projects (jobs):  
  $P = \{\text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}, \text{P06}, \text{P07}\}$

Let  
- $\text{skill}(w)$ = skill level of worker $w$ (Junior $<$ Intermediate $<$ Senior $<$ Expert)  
- $\text{required\_skill}(p)$ = required skill for project $p$  
- $\text{cost}_{w,p}$ = cost in cents for worker $w$ to do project $p$ (only for listed offers)  
- $E$ = set of eligible assignment pairs $(w,p)$:  
  - $w$ is not on leave  
  - $(w,p)$ is a listed offer  
  - $\text{skill}(w) \geq \text{required\_skill}(p)$ (using the order: Junior $<$ Intermediate $<$ Senior $<$ Expert)

##### Data

###### Worker data (on\_leave=0):

| worker_id | skill        |
|-----------|-------------|
| W00       | Junior      |
| W19       | Junior      |
| W11       | Senior      |
| W04       | Junior      |
| W15       | Expert      |
| W10       | Expert      |
| W06       | Expert      |
| W20       | Intermediate|
| W14       | Intermediate|

###### Project data:

| project_id | required_skill |
|------------|---------------|
| P00        | Intermediate  |
| P01        | Intermediate  |
| P02        | Senior        |
| P03        | Junior        |
| P04        | Junior        |
| P05        | Junior        |
| P06        | Junior        |
| P07        | Junior        |

###### Skill order mapping:

Junior = 1, Intermediate = 2, Senior = 3, Expert = 4

###### Offer list (only listed offers are permitted):

Below, only eligible offers are included (on\_leave=0 and skill(w) $\geq$ required\_skill(p)):

| worker_id | skill        | project_id | required_skill | cost_cents |
|-----------|-------------|------------|---------------|------------|
| W10       | Expert      | P00        | Intermediate  | 147        |
| W10       | Expert      | P01        | Intermediate  | 114        |
| W10       | Expert      | P03        | Junior        | 998        |
| W10       | Expert      | P04        | Junior        | 1270       |
| W10       | Expert      | P07        | Junior        | 953        |
| W19       | Junior      | P03        | Junior        | 109        |
| W19       | Junior      | P04        | Junior        | 1306       |
| W19       | Junior      | P05        | Junior        | 626        |
| W19       | Junior      | P06        | Junior        | 466        |
| W19       | Junior      | P07        | Junior        | 1365       |
| W11       | Senior      | P00        | Intermediate  | 235        |
| W11       | Senior      | P02        | Senior        | 1034       |
| W11       | Senior      | P03        | Junior        | 782        |
| W11       | Senior      | P04        | Junior        | 556        |
| W11       | Senior      | P05        | Junior        | 1018       |
| W11       | Senior      | P07        | Junior        | 1136       |
| W00       | Junior      | P04        | Junior        | 430        |
| W00       | Junior      | P05        | Junior        | 1385       |
| W00       | Junior      | P06        | Junior        | 260        |
| W00       | Junior      | P07        | Junior        | 1107       |
| W20       | Intermediate| P00        | Intermediate  | 675        |
| W20       | Intermediate| P01        | Intermediate  | 1055       |
| W20       | Intermediate| P03        | Junior        | 703        |
| W20       | Intermediate| P06        | Junior        | 893        |
| W20       | Intermediate| P07        | Junior        | 129        |
| W04       | Junior      | P03        | Junior        | 543        |
| W04       | Junior      | P04        | Junior        | 205        |
| W04       | Junior      | P05        | Junior        | 1149       |
| W04       | Junior      | P06        | Junior        | 533        |
| W04       | Junior      | P07        | Junior        | 732        |
| W06       | Expert      | P00        | Intermediate  | 1217       |
| W06       | Expert      | P01        | Intermediate  | 425        |
| W06       | Expert      | P02        | Senior        | 1320       |
| W06       | Expert      | P03        | Junior        | 1097       |
| W06       | Expert      | P04        | Junior        | 822        |
| W06       | Expert      | P05        | Junior        | 221        |
| W06       | Expert      | P06        | Junior        | 496        |
| W06       | Expert      | P07        | Junior        | 908        |
| W14       | Intermediate| P01        | Intermediate  | 1361       |
| W14       | Intermediate| P03        | Junior        | 379        |
| W14       | Intermediate| P04        | Junior        | 239        |
| W14       | Intermediate| P05        | Junior        | 460        |
| W14       | Intermediate| P06        | Junior        | 130        |
| W15       | Expert      | P00        | Intermediate  | 722        |
| W15       | Expert      | P02        | Senior        | 577        |
| W15       | Expert      | P03        | Junior        | 1062       |
| W15       | Expert      | P04        | Junior        | 1314       |
| W15       | Expert      | P05        | Junior        | 528        |
| W15       | Expert      | P06        | Junior        | 835        |
| W15       | Expert      | P07        | Junior        | 918        |

##### Decision Variables

For each eligible pair $(w,p) \in E$:

$$
x_{w,p} =
\begin{cases}
1 & \text{if worker } w \text{ is assigned to project } p \\
0 & \text{otherwise}
\end{cases}
$$

##### Objective Function

Minimize the total assignment cost (in cents):

$$
\min \sum_{(w,p) \in E} \text{cost}_{w,p} \cdot x_{w,p}
$$

##### Constraints

1. **Each project is assigned to exactly one worker:**

$$
\sum_{\substack{w: (w,p) \in E}} x_{w,p} = 1 \quad \forall p \in P
$$

2. **Each worker is assigned to at most one project:**

$$
\sum_{\substack{p: (w,p) \in E}} x_{w,p} \leq 1 \quad \forall w \in W
$$

3. **Eligibility:**

$$
x_{w,p} = 0 \quad \text{if } (w,p) \notin E
$$

4. **Variable domain:**

$$
x_{w,p} \in \{0,1\} \quad \forall (w,p) \in E
$$

##### Retrieved Information

{
  "workers": {
    "W00": {"skill": "Junior", "on_leave": 0},
    "W19": {"skill": "Junior", "on_leave": 0},
    "W11": {"skill": "Senior", "on_leave": 0},
    "W04": {"skill": "Junior", "on_leave": 0},
    "W15": {"skill": "Expert", "on_leave": 0},
    "W10": {"skill": "Expert", "on_leave": 0},
    "W06": {"skill": "Expert", "on_leave": 0},
    "W20": {"skill": "Intermediate", "on_leave": 0},
    "W14": {"skill": "Intermediate", "on_leave": 0}
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
    {"worker_id": "W10", "project_id": "P00", "cost_cents": 147},
    {"worker_id": "W10", "project_id": "P01", "cost_cents": 114},
    {"worker_id": "W10", "project_id": "P03", "cost_cents": 998},
    {"worker_id": "W10", "project_id": "P04", "cost_cents": 1270},
    {"worker_id": "W10", "project_id": "P07", "cost_cents": 953},
    {"worker_id": "W19", "project_id": "P03", "cost_cents": 109},
    {"worker_id": "W19", "project_id": "P04", "cost_cents": 1306},
    {"worker_id": "W19", "project_id": "P05", "cost_cents": 626},
    {"worker_id": "W19", "project_id": "P06", "cost_cents": 466},
    {"worker_id": "W19", "project_id": "P07", "cost_cents": 1365},
    {"worker_id": "W11", "project_id": "P00", "cost_cents": 235},
    {"worker_id": "W11", "project_id": "P02", "cost_cents": 1034},
    {"worker_id": "W11", "project_id": "P03", "cost_cents": 782},
    {"worker_id": "W11", "project_id": "P04", "cost_cents": 556},
    {"worker_id": "W11", "project_id": "P05", "cost_cents": 1018},
    {"worker_id": "W11", "project_id": "P07", "cost_cents": 1136},
    {"worker_id": "W00", "project_id": "P04", "cost_cents": 430},
    {"worker_id": "W00", "project_id": "P05", "cost_cents": 1385},
    {"worker_id": "W00", "project_id": "P06", "cost_cents": 260},
    {"worker_id": "W00", "project_id": "P07", "cost_cents": 1107},
    {"worker_id": "W20", "project_id": "P00", "cost_cents": 675},
    {"worker_id": "W20", "project_id": "P01", "cost_cents": 1055},
    {"worker_id": "W20", "project_id": "P03", "cost_cents": 703},
    {"worker_id": "W20", "project_id": "P06", "cost_cents": 893},
    {"worker_id": "W20", "project_id": "P07", "cost_cents": 129},
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
    {"worker_id": "W14", "project_id": "P01", "cost_cents": 1361},
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
    {"worker_id": "W15", "project_id": "P07", "cost_cents": 918}
  ]
}