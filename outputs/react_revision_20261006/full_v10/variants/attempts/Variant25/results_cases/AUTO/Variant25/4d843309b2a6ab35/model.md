##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Worker Assignment

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Leave Exclusion

$x_{w,p} = 0 \quad \forall w \in W_{\text{leave}}, \forall p \in P$

###### 4. Skill Feasibility

$x_{w,p} = 0 \quad \forall (w,p) \text{ where } \text{skill}(w) < \text{required\_skill}(p)$

###### 5. Forbidden Assignments

$x_{w,p} = 0 \quad \forall (w,p) \text{ where } c_{w,p} \text{ is blank in file_2_view_0}$

###### 6. Variable Domains

$x_{w,p} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker_id from file_0_view_0 with on_leave = 0
- $W_{\text{leave}}$: Set of worker_id from file_0_view_0 with on_leave = 1
- $P$: Set of project_id from file_1_view_0
- $\text{skill}(w)$: skill from file_0_view_0 for worker $w$
- $\text{required\_skill}(p)$: required_skill from file_1_view_0 for project $p$
- $c_{w,p}$: Value in file_2_view_0 at row worker_id $w$, column $p$ (project_id); blank cells forbid assignment
- $x_{w,p}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise

##### Notes

- Skill levels are ordered: Junior < Intermediate < Senior < Expert. A worker can only be assigned if their skill is at least the required_skill for the project.
- The objective minimizes the total assignment cost in USD cents, as given in $c_{w,p}$.
- Each project must be assigned to exactly one eligible worker; each eligible worker may be assigned at most one project. Workers on leave are excluded from all assignments. Blank cells in the cost matrix forbid assignment.