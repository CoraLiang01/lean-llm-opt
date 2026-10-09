##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \forall j \in \text{Projects}$

##### Data Mapping

- $c_{ij}$: Cost parameter for assigning manager $i$ to project $j$, from table_id "file_0_view_0", with manager index $i$ given by column "Unnamed: 0" and project index $j$ given by columns "P1", "P2", "P3", "P4", "P5", "P6".
- $\text{Managers}$: All unique values in column "Unnamed: 0" of table_id "file_0_view_0".
- $\text{Projects}$: All column names except "Unnamed: 0" in table_id "file_0_view_0".
- $x_{ij}$: Binary decision variable, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise.