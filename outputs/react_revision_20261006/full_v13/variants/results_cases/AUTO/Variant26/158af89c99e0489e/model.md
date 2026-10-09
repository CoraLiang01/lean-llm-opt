##### Objective Function:

$\min \sum_{(w,p) \in \mathcal{A}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

where $\mathcal{A}$ is the set of all (worker_id, project_id) pairs in the offer list that:
- have worker_id in the filtered worker set (on_leave = 0),
- and for which the worker's skill level is at least as high as the required_skill for the project (with Junior < Intermediate < Senior < Expert).

##### Constraints

###### 1. Project Assignment

$\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

Each project must be assigned to exactly one eligible worker.

###### 2. Worker Assignment

$\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

Each worker can be assigned to at most one project.

###### 3. Skill Feasibility

$x_{w,p} = 0$ if the skill level of worker $w$ is less than the required_skill of project $p$.

###### 4. Offer List Restriction

$x_{w,p} = 0$ unless (w,p) is present in the offer list.

###### 5. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}$

##### Data Mapping

- Worker set $\mathcal{W}$: All worker_id in file_0_view_0 (export_01.csv) with on_leave = 0.
- Project set $\mathcal{P}$: All project_id in file_1_view_0 (export_02.csv).
- Skill level for each worker: skill column in file_0_view_0.
- Required skill for each project: required_skill column in file_1_view_0.
- Offer list $\mathcal{A}$: All (worker_id, project_id) pairs in file_2_view_0 (export_03.csv) where worker_id $\in \mathcal{W}$.
- Assignment cost $\text{cost\_cents}_{w,p}$: cost_cents column in file_2_view_0 for each (worker_id, project_id) in $\mathcal{A}$.
- Skill hierarchy: Junior < Intermediate < Senior < Expert (as per user description).

##### Notes

- The minimum total cost is reported in USD cents (as per cost_cents).
- Only assignments present in the offer list and meeting all constraints are permitted.