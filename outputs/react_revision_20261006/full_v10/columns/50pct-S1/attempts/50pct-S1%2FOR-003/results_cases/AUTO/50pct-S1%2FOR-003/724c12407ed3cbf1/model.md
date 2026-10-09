##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Manager" of table_id file_0_view_0.
- Projects: All columns with names matching "Project * Cost" in table_id file_0_view_0.
- $c_{ij}$: Value in column "Project k Cost" for manager $i$ and project $j$ in table_id file_0_view_0, where $i$ is the row with "Manager" = manager name, and $j$ is the project corresponding to column "Project k Cost".
- $x_{ij}$: Binary variable indicating assignment of manager $i$ to project $j$.