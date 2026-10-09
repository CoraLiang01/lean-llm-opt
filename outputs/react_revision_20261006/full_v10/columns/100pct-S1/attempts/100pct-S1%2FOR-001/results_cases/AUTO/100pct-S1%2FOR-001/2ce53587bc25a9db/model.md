##### Objective Function:

$\min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

Each manager is assigned to exactly one project:
$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

Each project is assigned to exactly one manager:
$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

Variable domains:
$x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P$

##### Data Mapping

- $M$: set of managers, from column "Unnamed: 0" in table_id file_0_view_0
- $P$: set of projects, from columns "P1", "P2", "P3" in table_id file_0_view_0
- $c_{mp}$: cost parameter, from value at row with manager $m$ and column $p$ in table_id file_0_view_0
- $x_{mp}$: binary assignment variable, equals 1 if manager $m$ is assigned to project $p$, 0 otherwise

All sets and parameters are defined exactly as in the current CSV data.