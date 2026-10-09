##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$ &nbsp;&nbsp;&nbsp;&nbsp; (Each project is assigned exactly one worker)

$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$ &nbsp;&nbsp;&nbsp;&nbsp; (Each worker is assigned at most one project)

###### 2. Eligibility Constraints:

$x_{wp} = 0$ if:
- Worker $w$ is on leave (on_leave = 1 in file_0_view_0), or
- $c_{wp}$ is blank (no cost entry for $(w,p)$ in any of file_2_view_0, file_3_view_0, file_4_view_0), or
- Worker $w$'s skill (skill in file_0_view_0) is less than the required_skill for project $p$ (required_skill in file_1_view_0), using the order Junior < Intermediate < Senior < Expert.

###### 3. Variable Domains:

$x_{wp} \in \{0,1\} \quad \forall w \in W,\, p \in P$

##### Data Mapping

- $W$: Set of worker_id from file_0_view_0 with on_leave = 0.
- $P$: Set of project_id from file_1_view_0.
- $c_{wp}$: Cost in cents for worker $w$ to perform project $p$, taken from the cell for worker_id $w$ and project_id $p$ in the union of file_2_view_0, file_3_view_0, file_4_view_0. If multiple files contain a value for the same $(w,p)$, use the value from the file where $w$ appears.
- Worker skill: skill column in file_0_view_0.
- Project required_skill: required_skill column in file_1_view_0.
- Skill order: Junior < Intermediate < Senior < Expert.

##### Minimum Cost

The minimum total assignment cost is the optimal value of the objective function, in USD cents.