##### Objective Function:

$\quad \min \sum_{(w,p) \in \mathcal{A}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

##### Constraints

###### 1. Project Assignment:
$\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

###### 2. Worker Assignment:
$\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}_{\text{eligible}}$

###### 3. Skill Feasibility:
$x_{w,p} = 0 \quad \text{if } \text{skill}_w < \text{required\_skill}_p$

###### 4. Leave Exclusion:
$x_{w,p} = 0 \quad \text{if } w \notin \mathcal{W}_{\text{eligible}}$

###### 5. Offer List Restriction:
$x_{w,p} = 0 \quad \text{if } (w,p) \notin \mathcal{A}$

###### 6. Variable Domain:
$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}$

##### Index and Parameter Definitions

- $\mathcal{W}$: Set of all worker_id from file_0_view_0.
- $\mathcal{W}_{\text{eligible}}$: Subset of $\mathcal{W}$ with on_leave = 0.
- $\mathcal{P}$: Set of all project_id from file_1_view_0.
- $\mathcal{A}$: Set of all (worker_id, project_id) pairs listed in file_2_view_0, with worker_id $\in \mathcal{W}_{\text{eligible}}$.
- $\text{cost\_cents}_{w,p}$: Cost in cents for worker $w$ to do project $p$, from file_2_view_0.
- $\text{skill}_w$: Skill level of worker $w$ from file_0_view_0, ordered as Junior $<$ Intermediate $<$ Senior $<$ Expert.
- $\text{required\_skill}_p$: Required skill for project $p$ from file_1_view_0, ordered as above.

##### Data Mapping

{
  "workers": {
    "table_id": "file_0_view_0",
    "columns": ["worker_id", "skill", "on_leave"]
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

##### Notes

- The minimum assignment cost is reported in USD cents.
- Only assignments in $\mathcal{A}$ (the offer list) are permitted.
- Workers on leave (on_leave=1) are excluded from all assignments.
- Skill levels are compared ordinally: a worker can only be assigned if their skill is at least the required_skill for the project.