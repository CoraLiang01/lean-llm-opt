##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers},\; j \in \text{Projects}$

##### Data Mapping

- $c_{ij}$: Cost of assigning manager $i$ to project $j$, from table_id = "file_0_view_0", column "Manager" for $i$ and columns "Project 1 Cost", ..., "Project 11 Cost" for $j$.
- $\text{Managers}$: All unique values in column "Manager" of table_id = "file_0_view_0".
- $\text{Projects}$: All project columns "Project 1 Cost", ..., "Project 11 Cost" of table_id = "file_0_view_0", corresponding to project indices $j$.
- $x_{ij}$: Binary decision variable, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise.