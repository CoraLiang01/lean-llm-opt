##### Objective Function:

$\quad \min \sum_{w \in W} \sum_{p \in P} c_{w,p} \, x_{w,p}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{w \in W} x_{w,p} = 1 \quad \forall p \in P$  
$\sum_{p \in P} x_{w,p} \leq 1 \quad \forall w \in W$

###### 2. Leave and Skill Feasibility:

$x_{w,p} = 0$ if $\text{on\_leave}_{w} = 1$  
$x_{w,p} = 0$ if $\text{skill}_{w} \prec \text{required\_skill}_{p}$  
$x_{w,p} = 0$ if $c_{w,p}$ is blank

###### 3. Variable Domains:

$x_{w,p} \in \{0,1\} \quad \forall w \in W,\, p \in P$

##### Data Mapping

- $W$: All worker_id in file_0_view_0 with on_leave = 0
- $P$: All project_id in file_1_view_0
- $\text{on\_leave}_{w}$: from file_0_view_0, column "on_leave", indexed by worker_id
- $\text{skill}_{w}$: from file_0_view_0, column "skill", indexed by worker_id
- $\text{required\_skill}_{p}$: from file_1_view_0, column "required_skill", indexed by project_id
- $c_{w,p}$: assignment cost in cents, from the union of file_2_view_0, file_3_view_0, file_4_view_0, columns "P00"..."P09", indexed by worker_id and project_id; blank cells forbid assignment ($x_{w,p}=0$)
- Skill order: Junior $<$ Intermediate $<$ Senior $<$ Expert

- $x_{w,p}$: binary variable, 1 if worker $w$ is assigned to project $p$, 0 otherwise

##### Minimum Cost

- The minimum total assignment cost is reported in USD cents.