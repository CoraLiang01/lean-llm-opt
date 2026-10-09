##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

###### 3. Variable Domains:

$x_{mp} \in \{0,1\} \quad \forall m \in M, \; p \in P$

##### Data Mapping

- $M$: Set of managers, from column "Unnamed: 1" in table_id "file_0_view_0"
- $P$: Set of projects, from columns "P1", "P2", "P3" in table_id "file_0_view_0"
- $c_{mp}$: Cost for manager $m$ to manage project $p$, from the intersection of row with "Unnamed: 1" = $m$ and column $p$ in table_id "file_0_view_0"
- $x_{mp}$: Binary decision variable, equals 1 if manager $m$ is assigned to project $p$, 0 otherwise

All sets and parameters are defined exactly as in the current CSV data.