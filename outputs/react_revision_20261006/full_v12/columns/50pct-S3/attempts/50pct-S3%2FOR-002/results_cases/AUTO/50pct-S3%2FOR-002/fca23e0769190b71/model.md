##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

##### Constraints

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

$x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in P$

##### Data Mapping

- $M$: set of managers = {"MA", "MB", "MC", "MD", "ME", "MF"} (from column "Unnamed: 0" in table_id "file_0_view_0")
- $P$: set of projects = {"P1", "P2", "P3", "P4", "P5", "P6"} (from columns in table_id "file_0_view_0")
- $c_{ij}$: assignment cost of manager $i$ to project $j$ (from table_id "file_0_view_0", row with "Unnamed: 0" = $i$, column $j$)
- $x_{ij}$: binary variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise

All parameters and sets are mapped directly from "manager_project_costs.csv" (table_id "file_0_view_0").