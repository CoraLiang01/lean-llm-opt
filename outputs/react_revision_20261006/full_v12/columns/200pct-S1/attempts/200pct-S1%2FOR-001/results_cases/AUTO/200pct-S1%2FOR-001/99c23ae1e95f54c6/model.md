##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

$x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P$

##### Data Mapping

- $M$: set of managers, from column "Unnamed: 1" in table_id file_0_view_0
- $P$: set of projects, from columns ["P1", "P2", "P3"] in table_id file_0_view_0
- $c_{mp}$: cost for manager $m$ to complete project $p$, from table_id file_0_view_0, row indexed by manager $m$ ("Unnamed: 1"), column indexed by project $p$ ("P1", "P2", "P3")
- $x_{mp}$: binary variable, 1 if manager $m$ is assigned to project $p$, 0 otherwise

All data is sourced from table_id file_0_view_0 in "manager_project_costs.csv".