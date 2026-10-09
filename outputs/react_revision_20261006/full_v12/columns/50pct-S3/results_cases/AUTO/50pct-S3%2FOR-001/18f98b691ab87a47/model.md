##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

###### 3. Variable domains:

$x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P$

##### Data Mapping

- $M$: set of managers = all "Unnamed: 0" values in table_id file_0_view_0 of manager_project_costs.csv
- $P$: set of projects = all columns {"P1", "P2", "P3"} in table_id file_0_view_0 of manager_project_costs.csv
- $c_{mp}$: cost for manager $m$ to complete project $p$ = value at row with "Unnamed: 0" = $m$, column $p$ in table_id file_0_view_0 of manager_project_costs.csv
- $x_{mp}$: binary variable, 1 if manager $m$ is assigned to project $p$, 0 otherwise

All parameters and sets are defined exactly as in the source data.