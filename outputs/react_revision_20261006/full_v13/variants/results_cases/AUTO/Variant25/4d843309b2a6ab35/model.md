##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

###### 2. Worker Assignment

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

###### 3. Worker Availability

$x_{wp} = 0 \quad \forall w \in W: \text{on\_leave}_w = 1, \forall p \in P$

###### 4. Skill Feasibility

$x_{wp} = 0 \quad \forall (w,p): \text{skill}_w < \text{required\_skill}_p$

(With skill levels ordered: Junior < Intermediate < Senior < Expert)

###### 5. Assignment Feasibility

$x_{wp} = 0 \quad \forall (w,p): c_{wp} \text{ is blank in the cost matrix}$

###### 6. Variable Domain

$x_{wp} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker IDs from file_0_view_0 where on_leave = 0.
- $P$: Set of project IDs from file_1_view_0.
- $c_{wp}$: Assignment cost in USD cents from file_2_view_0, for worker $w$ and project $p$; blank cells forbid assignment.
- $\text{skill}_w$: Skill level of worker $w$ from file_0_view_0.
- $\text{required\_skill}_p$: Required skill for project $p$ from file_1_view_0.
- $x_{wp}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise.

- Skill ordering: Junior = 1, Intermediate = 2, Senior = 3, Expert = 4.

- All indices, costs, and attributes are mapped exactly as in:
  - Workers: file_0_view_0 (columns: worker_id, skill, on_leave)
  - Projects: file_1_view_0 (columns: project_id, required_skill)
  - Cost matrix: file_2_view_0 (rows: worker_id, columns: project_id)

- Only assignments with non-blank $c_{wp}$, on_leave = 0, and skill feasibility are allowed.

- The minimum total assignment cost is reported in USD cents.