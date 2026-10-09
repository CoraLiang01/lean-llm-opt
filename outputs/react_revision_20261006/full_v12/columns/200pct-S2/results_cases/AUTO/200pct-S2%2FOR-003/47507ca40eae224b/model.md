##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints:

Each manager is assigned to exactly one project:
$\quad \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

Each project is assigned to exactly one manager:
$\quad \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

Variable domains:
$\quad x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P$

##### Data Mapping

- $M$: Set of managers, from column "Manager" in table_id file_0_view_0 of manager_project_costs.csv
- $P$: Set of projects, from columns {"Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"} in table_id file_0_view_0
- $c_{mp}$: Cost parameter, from the value at row with "Manager" = $m$ and column = project cost column for $p$ in table_id file_0_view_0
- $x_{mp}$: Assignment variable, equals 1 if manager $m$ is assigned to project $p$, 0 otherwise

All sets and parameters are defined exactly as in the current source data.