##### Objective Function:

$\quad \min \sum_{i \in \text{Machines}} \sum_{j \in \text{Tasks}} c_{ij} x_{ij}$

##### Constraints

$\sum_{j \in \text{Tasks}} x_{ij} = 1 \quad \forall i \in \text{Machines}$

$\sum_{i \in \text{Machines}} x_{ij} = 1 \quad \forall j \in \text{Tasks}$

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Machines},\ j \in \text{Tasks}$

##### Data Mapping

- Machines: assignee_id from table_id file_0_view_0
- Tasks: assignment_cost_to_project_A, assignment_cost_to_project_B, ..., assignment_cost_to_project_L from table_id file_0_view_0
- $c_{ij}$: value in column assignment_cost_to_project_X for machine $i$ (row with assignee_id) and task $j$ (column assignment_cost_to_project_X), from table_id file_0_view_0
- $x_{ij}$: binary variable, 1 if machine $i$ is assigned to task $j$, 0 otherwise