##### Objective Function:

$\quad \min \sum_{(w,p) \in \mathcal{O}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

where $\mathcal{O}$ is the set of all (worker_id, project_id) pairs listed in the offer list (file_2_view_0) that satisfy:
- worker_id is present in file_0_view_0 (on_leave=0)
- worker's skill level $\geq$ required_skill for the project (using the order: Junior < Intermediate < Senior < Expert)

##### Constraints

###### 1. Project Assignment

$\sum_{\substack{w: (w,p) \in \mathcal{O}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

where $\mathcal{P}$ is the set of all project_id in file_1_view_0.

###### 2. Worker Assignment

$\sum_{\substack{p: (w,p) \in \mathcal{O}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

where $\mathcal{W}$ is the set of all worker_id in file_0_view_0.

###### 3. Skill Feasibility

$x_{w,p} = 0$ unless \text{skill}(w) $\geq$ \text{required\_skill}(p)$

(Skill levels are ordered: Junior=1, Intermediate=2, Senior=3, Expert=4.)

###### 4. Offer List Restriction

$x_{w,p} = 0$ unless $(w,p)$ is present in file_2_view_0.

###### 5. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{O}$

##### Data Mapping

{
  "workers": {
    "table_id": "file_0_view_0",
    "columns": ["worker_id", "skill"],
    "filter": {"on_leave": "0"}
  },
  "projects": {
    "table_id": "file_1_view_0",
    "columns": ["project_id", "required_skill"]
  },
  "offers": {
    "table_id": "file_2_view_0",
    "columns": ["worker_id", "project_id", "cost_cents"]
  },
  "skill_order": ["Junior", "Intermediate", "Senior", "Expert"]
}

##### Notes

- Only assignments $(w,p)$ present in the offer list (file_2_view_0) are permitted.
- For each assignment, the worker's skill must be at least the required_skill for the project, using the order above.
- Only workers with on_leave=0 (file_0_view_0) are eligible.
- The objective minimizes the total cost in USD cents.