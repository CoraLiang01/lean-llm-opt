##### Objective Function:

$\min \sum_{w \in W} \sum_{p \in P} c_{wp} \, x_{wp}$

##### Constraints

1. **Project Assignment:**  
$\sum_{w \in W} x_{wp} = 1 \quad \forall p \in P$

2. **Manager Assignment:**  
$\sum_{p \in P} x_{wp} \leq 1 \quad \forall w \in W$

3. **Eligibility:**  
$x_{wp} = 0$ if:
- Worker $w$ is on leave (on_leave = 1), or
- $c_{wp}$ is blank (assignment forbidden), or
- $\text{skill}(w) < \text{required\_skill}(p)$ (using Junior < Intermediate < Senior < Expert)

4. **Variable Domain:**  
$x_{wp} \in \{0,1\} \quad \forall w \in W,\, p \in P$

##### Data Mapping

- $W$: Set of worker_id from file_0_view_0 where on_leave = 0
- $P$: Set of project_id from file_1_view_0
- $\text{skill}(w)$: skill from file_0_view_0 for worker $w$
- $\text{required\_skill}(p)$: required_skill from file_1_view_0 for project $p$
- $c_{wp}$: entry in file_2_view_0, column $p$, row with worker_id $w$ (in USD cents); blank cells forbid assignment
- $x_{wp}$: binary assignment variable, 1 if worker $w$ assigned to project $p$, 0 otherwise

- Skill order: Junior < Intermediate < Senior < Expert

- Only $w$ with on_leave = 0 are eligible.

- For each $(w,p)$, $x_{wp}$ can be nonzero only if:
    - $c_{wp}$ is present (not blank)
    - $\text{skill}(w) \geq \text{required\_skill}(p)$ (using the above order)

- The minimum total cost is reported in USD cents.