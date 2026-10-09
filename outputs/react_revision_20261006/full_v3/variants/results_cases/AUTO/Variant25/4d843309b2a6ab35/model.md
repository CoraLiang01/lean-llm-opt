##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints

###### 1. Each project is assigned to exactly one eligible worker:

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Each worker is assigned to at most one project:

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Assignment only allowed if:
- Worker is not on leave,
- Worker skill $\geq$ project required_skill (with Junior < Intermediate < Senior < Expert),
- Assignment cost $c_{w,p}$ is defined (cell not blank):

$x_{w,p} = 0$ if any of the above is not satisfied.

###### 4. Variable domains:

$x_{w,p} \in \{0,1\} \quad \forall w \in W,\, p \in P$

##### Data Mapping

- $W$: All worker_id in file_0_view_0 with on_leave = 0.
- $P$: All project_id in file_1_view_0.
- $c_{w,p}$: Value in file_2_view_0 where worker_id = $w$ and column = $p$ (if not blank).
- Worker skill: file_0_view_0.skill for each worker_id.
- Project required_skill: file_1_view_0.required_skill for each project_id.
- Skill order: Junior < Intermediate < Senior < Expert (as per query).
- $x_{w,p}$: Binary assignment variable for worker $w$ to project $p$.

- Forbid $x_{w,p}$ if:
    - file_0_view_0.on_leave = 1 for $w$,
    - file_0_view_0.skill for $w$ is less than file_1_view_0.required_skill for $p$,
    - file_2_view_0 cell for ($w$, $p$) is blank.

- All costs $c_{w,p}$ are in USD cents.

##### Source Data Mapping

- Workers: file_0_view_0 (columns: worker_id, skill, on_leave)
- Projects: file_1_view_0 (columns: project_id, required_skill)
- Cost matrix: file_2_view_0 (columns: worker_id, P00, P01, P02, P03, P04, P05)
- Assignment variable: $x_{w,p}$ for $w \in W$, $p \in P$ as above.