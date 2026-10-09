##### Objective Function:

$\min \sum_{(w,p) \in \mathcal{A}} \text{cost\_cents}_{w,p} \cdot x_{w,p}$

##### Constraints

1. **Project Coverage:**  
$\sum_{\substack{w: (w,p) \in \mathcal{A}}} x_{w,p} = 1 \quad \forall p \in \mathcal{P}$

2. **Worker Assignment Limit:**  
$\sum_{\substack{p: (w,p) \in \mathcal{A}}} x_{w,p} \leq 1 \quad \forall w \in \mathcal{W}$

3. **Skill Feasibility:**  
$x_{w,p} = 0$ if $\text{skill}_w < \text{required\_skill}_p$ (using the order: Junior < Intermediate < Senior < Expert)

4. **Leave Exclusion:**  
$x_{w,p} = 0$ if $\text{on\_leave}_w = 1$

5. **Offer List Restriction:**  
$x_{w,p}$ is only defined for $(w,p) \in \mathcal{A}$, where $\mathcal{A}$ is the set of all $(w,p)$ pairs listed in the offer data.

6. **Variable Domain:**  
$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in \mathcal{A}$

##### Data Mapping

- $\mathcal{W}$: All worker_id in file_0_view_0 with on_leave = 0
- $\mathcal{P}$: All project_id in file_1_view_0
- $\text{skill}_w$: skill for worker_id $w$ from file_0_view_0
- $\text{on\_leave}_w$: on_leave for worker_id $w$ from file_0_view_0
- $\text{required\_skill}_p$: required_skill for project_id $p$ from file_1_view_0
- $\mathcal{A}$: All $(w,p)$ pairs in file_2_view_0 where worker_id $w$ and project_id $p$ are present, $w$ is not on leave, and $\text{skill}_w \geq \text{required\_skill}_p$ (using the skill order)
- $\text{cost\_cents}_{w,p}$: cost_cents for each $(w,p)$ from file_2_view_0

##### Notes

- The minimum total cost is reported in USD cents.
- The skill order is: Junior < Intermediate < Senior < Expert, and comparison is by this order.

##### Source Data Mapping

- file_0_view_0: [worker_id, skill, on_leave]
- file_1_view_0: [project_id, required_skill]
- file_2_view_0: [worker_id, project_id, cost_cents]