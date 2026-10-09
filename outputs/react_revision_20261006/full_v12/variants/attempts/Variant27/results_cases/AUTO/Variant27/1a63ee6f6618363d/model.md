##### Objective Function:

$\min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints:

1. **Project Assignment:**  
$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

2. **Worker Assignment:**  
$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

3. **Leave Exclusion:**  
$x_{w,p} = 0 \quad \forall w \in W: \text{on\_leave}_w = 1, \forall p \in P$

4. **Skill Feasibility:**  
$x_{w,p} = 0 \quad \forall (w,p) \in W \times P: \text{skill}_w < \text{required\_skill}_p$

5. **Assignment Feasibility (Cost Matrix):**  
$x_{w,p} = 0 \quad \forall (w,p) \in W \times P: c_{w,p} \text{ is blank}$

6. **Variable Domain:**  
$x_{w,p} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker IDs from `file_0_view_0` with $\text{on\_leave} = 0$.
- $P$: Set of project IDs from `file_1_view_0`.
- $\text{skill}_w$: Skill level of worker $w$ from `file_0_view_0` ("Junior", "Intermediate", "Senior", "Expert").
- $\text{required\_skill}_p$: Required skill for project $p$ from `file_1_view_0`.
- $c_{w,p}$: Assignment cost in USD cents for worker $w$ and project $p$ from the union of `file_2_view_0`, `file_3_view_0`, and `file_4_view_0` (columns: project IDs, rows: worker IDs). If blank, assignment is forbidden.
- $x_{w,p}$: Binary variable, $1$ if worker $w$ is assigned to project $p$, $0$ otherwise.

##### Source Tables

- Worker attributes: `file_0_view_0` (columns: worker_id, skill, on_leave)
- Project requirements: `file_1_view_0` (columns: project_id, required_skill)
- Assignment costs: `file_2_view_0`, `file_3_view_0`, `file_4_view_0` (columns: worker_id, P00–P09)

##### Skill Order

- "Junior" $<$ "Intermediate" $<$ "Senior" $<$ "Expert"

##### Objective

- Minimize the total assignment cost in USD cents.