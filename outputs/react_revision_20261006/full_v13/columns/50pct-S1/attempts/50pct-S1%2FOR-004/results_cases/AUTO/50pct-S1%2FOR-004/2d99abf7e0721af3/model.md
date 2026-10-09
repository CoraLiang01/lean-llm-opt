##### Objective Function:

$\quad \min \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{T}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \mathcal{T}} x_{ij} = 1 \quad \forall i \in \mathcal{M}$

$\sum_{i \in \mathcal{M}} x_{ij} = 1 \quad \forall j \in \mathcal{T}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{M},\; j \in \mathcal{T}$

##### Retrieved Information

{
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_ids": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"],
    "column_ids": [
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
    ]
  },
  "machines": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"],
  "tasks": [
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
  ]
}

Where:
- $\mathcal{M}$ is the set of machines (row IDs from "assignee_id" in table_id "file_0_view_0")
- $\mathcal{T}$ is the set of tasks (column IDs as listed above)
- $c_{ij}$ is the machining cost for assigning machine $i$ to task $j$, from the corresponding cell in the table
- $x_{ij}$ is a binary variable equal to 1 if machine $i$ is assigned to task $j$, 0 otherwise