##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Worker Assignment

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Worker Availability

$x_{w,p} = 0 \quad \forall w \in W_{\text{leave}}, \forall p \in P$

###### 4. Skill Feasibility

$x_{w,p} = 0 \quad \forall (w,p): \text{skill}(w) < \text{required\_skill}(p)$

###### 5. Assignment Feasibility (Cost Matrix)

$x_{w,p} = 0 \quad \forall (w,p): c_{w,p} \text{ is blank}$

###### 6. Variable Domain

$x_{w,p} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker IDs from file_0_view_0 where on_leave = 0, i.e., all worker_id with on_leave = 0 in /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_01/export_01.csv
- $W_{\text{leave}}$: Set of worker IDs from file_0_view_0 where on_leave = 1
- $P$: Set of project IDs from file_1_view_0, i.e., all project_id in /Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant25/inputs/batch_02/export_02.csv
- $\text{skill}(w)$: Skill level of worker $w$ from file_0_view_0, column skill
- $\text{required\_skill}(p)$: Required skill for project $p$ from file_1_view_0, column required_skill
- $c_{w,p}$: Assignment cost from file_2_view_0, row worker_id $w$, column $p$; blank cells forbid assignment
- $x_{w,p}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise

##### Skill Order

$\text{Junior} < \text{Intermediate} < \text{Senior} < \text{Expert}$

##### Objective Value

The minimum is the optimal value of the objective function, in USD cents.