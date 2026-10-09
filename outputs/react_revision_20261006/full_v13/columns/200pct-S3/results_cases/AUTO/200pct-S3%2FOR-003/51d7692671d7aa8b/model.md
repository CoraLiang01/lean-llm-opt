##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

where $c_{ij}$ is the cost for manager $i$ to complete project $j$, as given in the "manager_project_costs.csv" file (see Data Mapping).

##### Constraints

###### 1. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 2. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \forall j \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in the "Manager" column of table_id file_0_view_0.
- Projects: All columns with names "Project 1 Cost", "Project 2 Cost", ..., "Project 7 Cost" in table_id file_0_view_0.
- $c_{ij}$: The value in row with "Manager" = $i$ and column "Project $k$ Cost" = $j$ in table_id file_0_view_0, where $i$ and $j$ are as above.
- $x_{ij}$: Binary decision variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise.

All parameters and sets are defined exactly as in the source data.