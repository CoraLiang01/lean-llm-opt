##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Worker Assignment

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Worker Availability

$x_{w,p} = 0 \quad \forall w \in W: \text{on\_leave}_w = 1, \forall p \in P$

###### 4. Skill Feasibility

$x_{w,p} = 0 \quad \forall (w,p): \text{skill}_w < \text{required\_skill}_p$

(With skill levels ordered: Junior < Intermediate < Senior < Expert)

###### 5. Assignment Feasibility (Cost Matrix)

$x_{w,p} = 0 \quad \forall (w,p): c_{w,p} \text{ is blank in the cost matrix}$

###### 6. Variable Domains

$x_{w,p} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker IDs from file_0_view_0, with on_leave = 0
- $P$: Set of project IDs from file_1_view_0
- $c_{w,p}$: Assignment cost in USD cents from file_2_view_0, for worker $w$ and project $p$ (blank cells forbidden)
- $\text{on\_leave}_w$: From file_0_view_0, column "on_leave"
- $\text{skill}_w$: From file_0_view_0, column "skill"
- $\text{required\_skill}_p$: From file_1_view_0, column "required_skill"
- Skill order: Junior < Intermediate < Senior < Expert

- $x_{w,p}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise

- All indices, parameters, and constraints are mapped directly to the following tables:
    - Workers: table_id file_0_view_0, columns ["worker_id", "skill", "on_leave"]
    - Projects: table_id file_1_view_0, columns ["project_id", "required_skill"]
    - Cost matrix: table_id file_2_view_0, columns ["worker_id", "P00", "P01", "P02", "P03", "P04", "P05"]

- Only workers with on_leave = 0 are eligible.
- Only assignments where the worker's skill is at least the required_skill are eligible.
- Only assignments with a non-blank cost are eligible.

##### Objective

Minimize the total assignment cost in USD cents.