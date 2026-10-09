##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 3. Variable Domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers},\ \forall j \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Unnamed: 0" of table_id "file_0_view_0" in "manager_project_costs.csv"
- Projects: All columns except "Unnamed: 0" in table_id "file_0_view_0" in "manager_project_costs.csv"
- $c_{ij}$: Value at row with "Unnamed: 0" = $i$, column $j$ in table_id "file_0_view_0" in "manager_project_costs.csv"
- $x_{ij}$: Binary decision variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise