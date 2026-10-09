##### Objective Function:

$\quad \min \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{T}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each machine is assigned to exactly one task:

$\sum_{j \in \mathcal{T}} x_{ij} = 1 \quad \forall i \in \mathcal{M}$

###### 2. Each task is assigned to exactly one machine:

$\sum_{i \in \mathcal{M}} x_{ij} = 1 \quad \forall j \in \mathcal{T}$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{M}, \; j \in \mathcal{T}$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "machine_ids": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"],
    "task_ids": [
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
    "cost_matrix": "c_{ij} = \u2018assignment_cost_to_project_X\u2019 value for machine i (row with assignee_id M*) in table_id file_0_view_0"
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