##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

$x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P$

##### Data Mapping

- $M$: Set of managers, from column "Manager" in table_id file_0_view_0 of manager_project_costs.csv
- $P$: Set of projects, from columns {"Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"} in table_id file_0_view_0
- $c_{mp}$: Cost for manager $m$ to complete project $p$, from the intersection of row "Manager" $m$ and column $p$ in table_id file_0_view_0
- $x_{mp}$: Binary variable, 1 if manager $m$ is assigned to project $p$, 0 otherwise

All sets and parameters are defined exactly as in the source data.