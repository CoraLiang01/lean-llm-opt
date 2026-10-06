##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \forall j \in \text{Projects}$

##### Data Mapping

- $c_{ij}$: Cost for manager $i$ to complete project $j$, from table_id: file_0_view_0, columns: "Unnamed: 0" (manager), "P1", "P2", "P3" (projects).
- $\text{Managers}$: All unique values in column "Unnamed: 0" of table_id: file_0_view_0.
- $\text{Projects}$: All column names except "Unnamed: 0" in table_id: file_0_view_0.