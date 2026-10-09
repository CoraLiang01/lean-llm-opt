##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

where $c_{ij}$ is the cost of assigning manager $i$ to project $j$, as given in the data mapping below.

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

##### Data Mapping

- $M$ (Managers): All unique values in column "Unnamed: 0" of table_id file_0_view_0 in "manager_project_costs.csv"
- $P$ (Projects): All columns ["P1", "P2", "P3", "P4", "P5", "P6"] of table_id file_0_view_0 in "manager_project_costs.csv"
- $c_{ij}$: Value at row with "Unnamed: 0" = $i$ and column $j$ in table_id file_0_view_0 in "manager_project_costs.csv"
- $x_{ij}$: Binary decision variable, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise

All sets and parameters are defined exactly as in the current CSV data.