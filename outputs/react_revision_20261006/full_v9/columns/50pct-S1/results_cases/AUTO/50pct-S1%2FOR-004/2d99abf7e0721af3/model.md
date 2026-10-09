##### Objective Function:

$\quad \min \sum_{i \in \text{Machines}} \sum_{j \in \text{Tasks}} c_{ij} x_{ij}$

##### Constraints

###### 1. Each machine is assigned to exactly one task:

$\sum_{j \in \text{Tasks}} x_{ij} = 1 \quad \forall i \in \text{Machines}$

###### 2. Each task is assigned to exactly one machine:

$\sum_{i \in \text{Machines}} x_{ij} = 1 \quad \forall j \in \text{Tasks}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Machines},\ \forall j \in \text{Tasks}$

##### Retrieved Information

{
  "Machines": [
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"
  ],
  "Tasks": [
    "assignment_cost_to_project_A",
    "assignment_cost_to_project_B",
    "assignment_cost_to_project_C",
    "assignment_cost_to_project_D",
    "assignment_cost_to_project_E",
    "assignment_cost_to_project_F",
    "assignment_cost_to_project_G",
    "assignment_cost_to_project_H",
    "assignment_cost_to_project_I",
    "assignment_cost_to_project_J",
    "assignment_cost_to_project_K",
    "assignment_cost_to_project_L"
  ],
  "Cost Parameter Mapping": {
    "table_id": "file_0_view_0",
    "row_index_set": "Machines (assignee_id)",
    "column_index_set": "Tasks (assignment_cost_to_project_*)",
    "parameter": "c_{ij} = [value in column j for row i]"
  }
}