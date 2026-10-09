##### Objective Function:

$\quad \min \sum_{(w,p) \in \mathcal{A}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

##### Constraints

###### 1. Project Assignment

$\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

###### 2. Worker Assignment (at most one project per worker)

$\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}_{\text{avail}}$

###### 3. Skill Feasibility

$x_{w,p} = 0$ if $\text{skill}_w < \text{required\_skill}_p$ for $(w,p) \in \mathcal{A}$

###### 4. Leave Exclusion

$x_{w,p} = 0$ if $w \notin \mathcal{W}_{\text{avail}}$ for $(w,p) \in \mathcal{A}$

###### 5. Offer List Restriction

$x_{w,p}$ is defined only for $(w,p) \in \mathcal{A}$ (the set of listed offers in file_2_view_0)

###### 6. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}$

##### Data Mapping

- $\mathcal{W}$: All worker_id in file_0_view_0
- $\mathcal{W}_{\text{avail}}$: All worker_id in file_0_view_0 with on_leave = 0
- $\mathcal{P}$: All project_id in file_1_view_0
- $\mathcal{A}$: All (worker_id, project_id) pairs in file_2_view_0
- $\text{cost\_cents}_{w,p}$: cost_cents from file_2_view_0 for each (worker_id, project_id)
- $\text{skill}_w$: skill from file_0_view_0 for each worker_id
- $\text{required\_skill}_p$: required_skill from file_1_view_0 for each project_id
- Skill order: Junior < Intermediate < Senior < Expert

- $x_{w,p}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise

- The minimum total cost in USD cents is the optimal value of the objective function.