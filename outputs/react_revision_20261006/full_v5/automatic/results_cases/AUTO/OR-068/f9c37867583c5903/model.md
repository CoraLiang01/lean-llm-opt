##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

###### Data Mapping

- $M$: Set of managers, from column "Unnamed: 0" in table_id "file_0_view_0" of "manager_project_costs.csv"
- $P$: Set of projects, from columns ["P1", "P2", "P3", "P4", "P5", "P6"] in table_id "file_0_view_0" of "manager_project_costs.csv"
- $c_{ij}$: Cost of assigning manager $i$ to project $j$, from the entry at row with "Unnamed: 0" = $i$ and column $j$ in table_id "file_0_view_0" of "manager_project_costs.csv"
- $x_{ij}$: Binary variable, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise, for all $i \in M$, $j \in P$