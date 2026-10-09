##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

where $c_{wp}$ is the assignment cost in USD cents for worker $w$ to project $p$, and $x_{wp}$ is a binary variable indicating assignment.

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

(Each project is assigned exactly one worker.)

###### 2. Worker Assignment

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

(Each worker is assigned to at most one project.)

###### 3. Leave Exclusion

$x_{wp} = 0 \quad \forall w \in W_{\text{leave}}, \forall p \in P$

(Workers with on_leave=1 cannot be assigned.)

###### 4. Skill Feasibility

$x_{wp} = 0 \quad \forall (w,p): \text{skill}(w) < \text{required\_skill}(p)$

(Worker's skill must meet or exceed the project's required_skill, with Junior < Intermediate < Senior < Expert.)

###### 5. Assignment Feasibility

$x_{wp} = 0 \quad \forall (w,p): c_{wp} \text{ is blank}$

(Blank cost cells forbid assignment.)

###### 6. Variable Domain

$x_{wp} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker_id from file_0_view_0 where on_leave = 0.
- $P$: Set of project_id from file_1_view_0.
- $c_{wp}$: Cost in USD cents from the union of file_2_view_0, file_3_view_0, file_4_view_0, indexed by (worker_id, project_id); blank cells are forbidden assignments.
- $\text{skill}(w)$: Skill level for worker $w$ from file_0_view_0.
- $\text{required\_skill}(p)$: Required skill for project $p$ from file_1_view_0.
- $W_{\text{leave}}$: Set of worker_id from file_0_view_0 where on_leave = 1.

All index sets, parameters, and constraints are defined exactly as in the retrieved data.