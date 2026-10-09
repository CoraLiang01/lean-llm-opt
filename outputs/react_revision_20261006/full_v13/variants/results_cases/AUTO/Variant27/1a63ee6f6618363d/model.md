##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where $c_{wp}$ is the assignment cost in USD cents for worker $w$ to project $p$, as given in the cost matrices (see Data Mapping). $x_{wp}$ is a binary variable indicating assignment.

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

(Each project is assigned exactly one worker.)

###### 2. Worker Assignment

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

(Each worker is assigned to at most one project.)

###### 3. Leave Exclusion

$x_{wp} = 0 \quad \forall w \in W_{\text{leave}}, \forall p \in P$

(Workers with on_leave=1 are excluded from all assignments.)

###### 4. Skill Feasibility

$x_{wp} = 0 \quad \forall (w,p): \text{skill}(w) < \text{required\_skill}(p)$

(Worker $w$ can only be assigned to project $p$ if their skill level is at least the required_skill for $p$, using the order: Junior < Intermediate < Senior < Expert.)

###### 5. Assignment Feasibility

$x_{wp} = 0 \quad \forall (w,p): c_{wp} \text{ is blank}$

(Assignments with blank cost cells are forbidden.)

###### 6. Variable Domain

$x_{wp} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker_id from file_0_view_0, excluding any with on_leave=1.
- $P$: Set of project_id from file_1_view_0.
- $c_{wp}$: Assignment cost in USD cents, from the union of cost matrices in file_2_view_0, file_3_view_0, and file_4_view_0, matched by worker_id and project_id. If the cell is blank, assignment is forbidden.
- $\text{skill}(w)$: Skill level for worker $w$ from file_0_view_0.
- $\text{required\_skill}(p)$: Required skill for project $p$ from file_1_view_0.
- $W_{\text{leave}}$: Set of worker_id with on_leave=1 in file_0_view_0.

##### Source Tables

- file_0_view_0: [worker_id, skill, on_leave]
- file_1_view_0: [project_id, required_skill]
- file_2_view_0, file_3_view_0, file_4_view_0: [worker_id, P00, P01, ..., P09] (cost matrices)

All indices, parameters, and constraints are defined using the above data. The minimum total assignment cost is reported in USD cents.