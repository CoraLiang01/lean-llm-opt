##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

$x_{mp} \in \{0,1\} \quad \forall m \in M, \; p \in P$

##### Data Mapping

- $M$: Set of managers, from "manager_project_costs.csv", column "Unnamed: 0", table_id "file_0_view_0"
- $P$: Set of projects, from "manager_project_costs.csv", columns ["P1", "P2", "P3"], table_id "file_0_view_0"
- $c_{mp}$: Cost for manager $m$ to complete project $p$, from "manager_project_costs.csv", table_id "file_0_view_0", row indexed by $m$, column indexed by $p$
- $x_{mp}$: Binary assignment variable, equals 1 if manager $m$ is assigned to project $p$, 0 otherwise