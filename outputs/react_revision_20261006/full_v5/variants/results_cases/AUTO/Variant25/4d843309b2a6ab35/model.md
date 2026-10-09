##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints

###### 1. Project Assignment

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$

###### 2. Worker Assignment

$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 3. Eligibility Constraints

$x_{w,p} = 0$ if:
- Worker $w$ is excluded (on_leave=1), or
- $c_{w,p}$ is blank (assignment forbidden), or
- $\text{skill}(w) < \text{required\_skill}(p)$ (with Junior < Intermediate < Senior < Expert)

###### 4. Variable Domains

$x_{w,p} \in \{0,1\} \quad \forall w \in W,\, p \in P$

##### Data Mapping

- $W$: Set of eligible workers (from file_0_view_0, where on_leave=0)
- $P$: Set of projects (from file_1_view_0, project_id)
- $\text{skill}(w)$: Worker skill from file_0_view_0, column "skill"
- $\text{required\_skill}(p)$: Project required_skill from file_1_view_0
- $c_{w,p}$: Assignment cost in USD cents from file_2_view_0, row worker_id $w$, column $p$ (project_id); blank cells forbid assignment
- $x_{w,p}$: Binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise

- Skill order: Junior < Intermediate < Senior < Expert

- All indices, parameters, and eligibility are defined by the CSV data as mapped above.