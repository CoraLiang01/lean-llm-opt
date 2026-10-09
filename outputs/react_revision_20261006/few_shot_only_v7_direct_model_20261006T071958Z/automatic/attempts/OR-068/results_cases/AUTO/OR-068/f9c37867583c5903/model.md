##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints:

1. **Manager Assignment:**  
$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

2. **Project Assignment:**  
$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

3. **Variable Domains:**  
$x_{ij} \in \{0,1\} \quad \forall i \in M, \; j \in P$

##### Data Mapping

- $M$ (Managers): All values in column "Unnamed: 0" of table_id "file_0_view_0"
- $P$ (Projects): All columns except "Unnamed: 0" in table_id "file_0_view_0"
- $c_{ij}$: Value at row with "Unnamed: 0" = $i$, column $j$ in table_id "file_0_view_0"
- $x_{ij}$: Binary decision variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise