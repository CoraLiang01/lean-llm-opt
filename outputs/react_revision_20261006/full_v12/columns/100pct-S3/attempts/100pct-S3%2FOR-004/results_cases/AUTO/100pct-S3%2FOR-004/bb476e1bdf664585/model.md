##### Objective Function:

$\quad \min \sum_{i \in \text{Machines}} \sum_{j \in \text{Tasks}} c_{ij} x_{ij}$

##### Constraints:

$\sum_{j \in \text{Tasks}} x_{ij} = 1 \quad \forall i \in \text{Machines}$

$\sum_{i \in \text{Machines}} x_{ij} = 1 \quad \forall j \in \text{Tasks}$

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Machines},\ j \in \text{Tasks}$

##### Data Mapping

- Machines: all assignee_id in column "assignee_id" of table_id file_0_view_0
- Tasks: all columns {"assignment_cost_to_project_A", "assignment_cost_to_project_B", "assignment_cost_to_project_C", "assignment_cost_to_project_D", "assignment_cost_to_project_E", "assignment_cost_to_project_F", "assignment_cost_to_project_G", "assignment_cost_to_project_H", "assignment_cost_to_project_I", "assignment_cost_to_project_J", "assignment_cost_to_project_K", "assignment_cost_to_project_L"} in table_id file_0_view_0
- $c_{ij}$: value in row with assignee_id $i$ and column $j$ in table_id file_0_view_0
- $x_{ij}$: binary variable, 1 if machine $i$ is assigned to task $j$, 0 otherwise