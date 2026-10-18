##### Objective Function:

$\quad \min \sum_{(w,p) \in \mathcal{A}} c_{w,p} \, x_{w,p}$

where $x_{w,p} = 1$ if worker $w$ is assigned to project $p$, $0$ otherwise, and $\mathcal{A}$ is the set of eligible and listed (worker, project) pairs.

##### Constraints

###### 1. Project Coverage

$\sum_{w: (w,p) \in \mathcal{A}} x_{w,p} = 1 \quad \forall p \in \{P00, P01, P02, P03, P04, P05, P06, P07\}$

###### 2. Worker Assignment (at most one project per worker)

$\sum_{p: (w,p) \in \mathcal{A}} x_{w,p} \leq 1 \quad \forall w \in \{W00, W04, W06, W10, W11, W14, W15, W19, W20\}$

###### 3. Eligibility Mask

$x_{w,p} = 0$ for any $(w,p)$ not in $\mathcal{A}$ (see below for $\mathcal{A}$).

###### 4. Variable Domain

$x_{w,p} \in \{0,1\}$ for all $(w,p) \in \mathcal{A}$

---

##### Eligibility and Offer Mask $\mathcal{A}$

A pair $(w,p)$ is in $\mathcal{A}$ if:

- The offer $(w,p)$ is listed in the data (see below).
- Worker $w$ is not on leave ($on\_leave = 0$).
- Worker $w$'s skill is at least as high as the required skill for project $p$ (Junior < Intermediate < Senior < Expert).

Skill order: Junior = 1, Intermediate = 2, Senior = 3, Expert = 4.

###### Worker Data

| worker_id | skill        | skill_level | on_leave |
|-----------|-------------|-------------|----------|
| W00       | Junior      | 1           | 0        |
| W04       | Junior      | 1           | 0        |
| W06       | Expert      | 4           | 0        |
| W10       | Expert      | 4           | 0        |
| W11       | Senior      | 3           | 0        |
| W14       | Intermediate| 2           | 0        |
| W15       | Expert      | 4           | 0        |
| W19       | Junior      | 1           | 0        |
| W20       | Intermediate| 2           | 0        |

(W01 is excluded: on_leave=1.)

###### Project Data

| project_id | required_skill | skill_level |
|------------|---------------|-------------|
| P00        | Intermediate  | 2           |
| P01        | Intermediate  | 2           |
| P02        | Senior        | 3           |
| P03        | Junior        | 1           |
| P04        | Junior        | 1           |
| P05        | Junior        | 1           |
| P06        | Junior        | 1           |
| P07        | Junior        | 1           |

###### Offer List (Eligible Pairs Only)

Below, only pairs where worker's skill $\geq$ required skill and worker is not on leave are included.

| worker_id | project_id | cost_cents | worker_skill | required_skill | Eligible? |
|-----------|------------|------------|-------------|---------------|-----------|
| W00       | P01        | 17         | Junior      | Intermediate  | No        |
| W00       | P04        | 430        | Junior      | Junior        | Yes       |
| W00       | P05        | 1385       | Junior      | Junior        | Yes       |
| W00       | P06        | 260        | Junior      | Junior        | Yes       |
| W00       | P07        | 1107       | Junior      | Junior        | Yes       |
| W04       | P00        | 4          | Junior      | Intermediate  | No        |
| W04       | P01        | 8          | Junior      | Intermediate  | No        |
| W04       | P02        | 16         | Junior      | Senior        | No        |
| W04       | P03        | 543        | Junior      | Junior        | Yes       |
| W04       | P04        | 205        | Junior      | Junior        | Yes       |
| W04       | P05        | 1149       | Junior      | Junior        | Yes       |
| W04       | P06        | 533        | Junior      | Junior        | Yes       |
| W04       | P07        | 732        | Junior      | Junior        | Yes       |
| W06       | P00        | 1217       | Expert      | Intermediate  | Yes       |
| W06       | P01        | 425        | Expert      | Intermediate  | Yes       |
| W06       | P02        | 1320       | Expert      | Senior        | Yes       |
| W06       | P03        | 1097       | Expert      | Junior        | Yes       |
| W06       | P04        | 822        | Expert      | Junior        | Yes       |
| W06       | P05        | 221        | Expert      | Junior        | Yes       |
| W06       | P06        | 496        | Expert      | Junior        | Yes       |
| W06       | P07        | 908        | Expert      | Junior        | Yes       |
| W10       | P00        | 147        | Expert      | Intermediate  | Yes       |
| W10       | P01        | 114        | Expert      | Intermediate  | Yes       |
| W10       | P03        | 998        | Expert      | Junior        | Yes       |
| W10       | P04        | 1270       | Expert      | Junior        | Yes       |
| W10       | P07        | 953        | Expert      | Junior        | Yes       |
| W11       | P00        | 235        | Senior      | Intermediate  | Yes       |
| W11       | P02        | 1034       | Senior      | Senior        | Yes       |
| W11       | P03        | 782        | Senior      | Junior        | Yes       |
| W11       | P04        | 556        | Senior      | Junior        | Yes       |
| W11       | P05        | 1018       | Senior      | Junior        | Yes       |
| W11       | P07        | 1136       | Senior      | Junior        | Yes       |
| W14       | P01        | 1361       | Intermediate| Intermediate  | Yes       |
| W14       | P03        | 379        | Intermediate| Junior        | Yes       |
| W14       | P04        | 239        | Intermediate| Junior        | Yes       |
| W14       | P05        | 460        | Intermediate| Junior        | Yes       |
| W14       | P06        | 130        | Intermediate| Junior        | Yes       |
| W15       | P00        | 722        | Expert      | Intermediate  | Yes       |
| W15       | P02        | 577        | Expert      | Senior        | Yes       |
| W15       | P03        | 1062       | Expert      | Junior        | Yes       |
| W15       | P04        | 1314       | Expert      | Junior        | Yes       |
| W15       | P05        | 528        | Expert      | Junior        | Yes       |
| W15       | P06        | 835        | Expert      | Junior        | Yes       |
| W15       | P07        | 918        | Expert      | Junior        | Yes       |
| W19       | P00        | 5          | Junior      | Intermediate  | No        |
| W19       | P02        | 9          | Junior      | Senior        | No        |
| W19       | P03        | 109        | Junior      | Junior        | Yes       |
| W19       | P04        | 1306       | Junior      | Junior        | Yes       |
| W19       | P05        | 626        | Junior      | Junior        | Yes       |
| W19       | P06        | 466        | Junior      | Junior        | Yes       |
| W19       | P07        | 1365       | Junior      | Junior        | Yes       |
| W20       | P00        | 675        | Intermediate| Intermediate  | Yes       |
| W20       | P01        | 1055       | Intermediate| Intermediate  | Yes       |
| W20       | P02        | 8          | Intermediate| Senior        | No        |
| W20       | P03        | 703        | Intermediate| Junior        | Yes       |
| W20       | P06        | 893        | Intermediate| Junior        | Yes       |
| W20       | P07        | 129        | Intermediate| Junior        | Yes       |

$\mathcal{A}$ is the set of all pairs above with "Eligible? = Yes".

##### Cost Matrix $c_{w,p}$

For each $(w,p) \in \mathcal{A}$, $c_{w,p}$ is the corresponding cost_cents.

---

##### Retrieved Information

```json
{
  "workers": {
    "W00": {"skill": "Junior", "skill_level": 1, "on_leave": 0},
    "W04": {"skill": "Junior", "skill_level": 1, "on_leave": 0},
    "W06": {"skill": "Expert", "skill_level": 4, "on_leave": 0},
    "W10": {"skill": "Expert", "skill_level": 4, "on_leave": 0},
    "W11": {"skill": "Senior", "skill_level": 3, "on_leave": 0},
    "W14": {"skill": "Intermediate", "skill_level": 2, "on_leave": 0},
    "W15": {"skill": "Expert", "skill_level": 4, "on_leave": 0},
    "W19": {"skill": "Junior", "skill_level": 1, "on_leave": 0},
    "W20": {"skill": "Intermediate", "skill_level": 2, "on_leave": 0}
  },
  "projects": {
    "P00": {"required_skill": "Intermediate", "skill_level": 2},
    "P01": {"required_skill": "Intermediate", "skill_level": 2},
    "P02": {"required_skill": "Senior", "skill_level": 3},
    "P03": {"required_skill": "Junior", "skill_level": 1},
    "P04": {"required_skill": "Junior", "skill_level": 1},
    "P05": {"required_skill": "Junior", "skill_level": 1},
    "P06": {"required_skill": "Junior", "skill_level": 1},
    "P07": {"required_skill": "Junior", "skill_level": 1}
  },
  "offers": [
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
    {"worker_id": "W15", "project_id": "P07", "cost_cents": 918},
    {"worker_id": "W19", "project_id": "P03", "cost_cents": 109},
    {"worker_id": "W19", "project_id": "P04", "cost_cents": 1306},
    {"worker_id": "W19", "project_id": "P05", "cost_cents": 626},
    {"worker_id": "W19", "project_id": "P06", "cost_cents": 466},
    {"worker_id": "W19", "project_id": "P07", "cost_cents": 1365},
    {"worker_id": "W20", "project_id": "P00", "cost_cents": 675},
    {"worker_id": "W20", "project_id": "P01", "cost_cents": 1055},
    {"worker_id": "W20", "project_id": "P03", "cost_cents": 703},
    {"worker_id": "W20", "project_id": "P06", "cost_cents": 893},
    {"worker_id": "W20", "project_id": "P07", "cost_cents": 129}
  ]
}
```

---

##### Summary

- Assign each project exactly one eligible worker (from the listed offers, not on leave, with sufficient skill).
- Each worker may be assigned at most one project.
- Only listed offers are permitted.
- Minimize total cost in USD cents.

---

##### Mathematical Model

$\boxed{
\begin{align*}
\min \quad & \sum_{(w,p) \in \mathcal{A}} c_{w,p} \, x_{w,p} \\
\text{s.t.} \quad & \sum_{w: (w,p) \in \mathcal{A}} x_{w,p} = 1 \quad \forall p \in \{P00, P01, P02, P03, P04, P05, P06, P07\} \\
& \sum_{p: (w,p) \in \mathcal{A}} x_{w,p} \leq 1 \quad \forall w \in \{W00, W04, W06, W10, W11, W14, W15, W19, W20\} \\
& x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}
\end{align*}
}$

where $\mathcal{A}$ and $c_{w,p}$ are as listed above.