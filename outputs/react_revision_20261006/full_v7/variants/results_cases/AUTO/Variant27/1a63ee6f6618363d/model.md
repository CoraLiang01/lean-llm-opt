##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

###### 2. Worker Assignment

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

###### 3. Leave Exclusion

$x_{wp} = 0 \quad \forall w \in W_{\text{leave}}, \forall p \in P$

###### 4. Skill Feasibility

$x_{wp} = 0 \quad \forall (w,p) \text{ where } \text{skill}(w) < \text{required\_skill}(p)$

###### 5. Assignment Feasibility

$x_{wp} = 0 \quad \forall (w,p) \text{ where } c_{wp} \text{ is blank}$

###### 6. Variable Domain

$x_{wp} \in \{0,1\} \quad \forall w \in W, \forall p \in P$

##### Data Mapping

- $W$: Set of worker IDs from file_0_view_0, excluding any with on_leave=1.
- $P$: Set of project IDs from file_1_view_0.
- $c_{wp}$: Assignment cost in USD cents for worker $w$ and project $p$, from the merged cost matrices (file_2_view_0, file_3_view_0, file_4_view_0), indexed by worker_id and project_id. If the cell is blank, assignment is forbidden.
- $x_{wp}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise.
- $\text{skill}(w)$: Skill level of worker $w$ from file_0_view_0.
- $\text{required\_skill}(p)$: Required skill for project $p$ from file_1_view_0.
- $W_{\text{leave}}$: Set of worker IDs with on_leave=1 in file_0_view_0.

##### Skill Order

- Junior < Intermediate < Senior < Expert

##### Index Sets

- $W = \{\text{worker\_id} \mid \text{on\_leave}=0 \text{ in file\_0\_view\_0}\}$
- $P = \{\text{project\_id} \text{ in file\_1\_view\_0}\}$

##### Cost Matrix

- $c_{wp}$ is defined only for $(w,p)$ pairs where a non-blank value exists in the cost matrix (file_2_view_0, file_3_view_0, file_4_view_0), matched by worker_id and project_id.

##### Variables

- $x_{wp}$: Binary assignment variable for each feasible $(w,p)$.

##### Minimum Cost

- The minimum total assignment cost is the optimal value of the objective function, in USD cents.