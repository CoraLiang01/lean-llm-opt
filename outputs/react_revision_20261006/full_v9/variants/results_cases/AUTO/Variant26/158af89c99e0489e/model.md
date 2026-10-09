##### Objective Function:

$\min \sum_{(w,p) \in \mathcal{A}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

##### Constraints

###### 1. Project Assignment

$\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

###### 2. Worker Assignment (at most one project per worker)

$\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

###### 3. Skill Feasibility

$x_{w,p} = 0$ if $\text{skill}_w < \text{required\_skill}_p$ (with Junior < Intermediate < Senior < Expert, as ordered)

###### 4. Leave Exclusion

$x_{w,p} = 0$ if $\text{on\_leave}_w = 1$

###### 5. Offer List Restriction

$x_{w,p}$ is only defined for $(w,p) \in \mathcal{A}$, where $\mathcal{A}$ is the set of all (worker_id, project_id) pairs listed in the offer table (file_2_view_0).

###### 6. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}$

##### Data Mapping

- $\mathcal{W}$: All worker_id in file_0_view_0 with on_leave = 0
- $\mathcal{P}$: All project_id in file_1_view_0
- $\mathcal{A}$: All (worker_id, project_id) pairs in file_2_view_0 where worker_id $\in \mathcal{W}$, and worker's skill $\geq$ required_skill for the project (using the order Junior < Intermediate < Senior < Expert)
- $\text{cost\_cents}_{w,p}$: cost_cents from file_2_view_0 for each $(w,p) \in \mathcal{A}$
- $\text{skill}_w$: skill from file_0_view_0 for each worker_id
- $\text{required\_skill}_p$: required_skill from file_1_view_0 for each project_id
- $\text{on\_leave}_w$: on_leave from file_0_view_0 for each worker_id

##### Notes

- The minimum is reported in USD cents (as per cost_cents).
- Only assignments present in the offer list (file_2_view_0) are permitted.
- Skill comparison uses the order: Junior < Intermediate < Senior < Expert.