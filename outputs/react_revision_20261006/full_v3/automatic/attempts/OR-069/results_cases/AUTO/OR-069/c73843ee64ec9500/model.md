##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Data Mapping

- $i \in$ Managers: all values in column "Manager" of table_id file_0_view_0 from "manager_project_costs.csv"
- $j \in$ Projects: all columns ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost", "Project 8 Cost", "Project 9 Cost", "Project 10 Cost", "Project 11 Cost"] of table_id file_0_view_0 from "manager_project_costs.csv"
- $c_{ij}$: value at row with "Manager" = $i$ and column = $j$ in table_id file_0_view_0 from "manager_project_costs.csv"
- $x_{ij}$: binary decision variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise