##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers},\ \forall j \in \text{Projects}$

##### Data Mapping

- Managers: All unique values in column "Unnamed: 0" of table_id "file_0_view_0" in "manager_project_costs.csv"
- Projects: All columns ["P1", "P2", "P3"] of table_id "file_0_view_0" in "manager_project_costs.csv"
- Cost parameter $c_{ij}$: Value in column $j$ for row with "Unnamed: 0" = $i$ in table_id "file_0_view_0" in "manager_project_costs.csv"
- Decision variable $x_{ij}$: 1 if manager $i$ is assigned to project $j$, 0 otherwise, for all $i$ in Managers and $j$ in Projects.