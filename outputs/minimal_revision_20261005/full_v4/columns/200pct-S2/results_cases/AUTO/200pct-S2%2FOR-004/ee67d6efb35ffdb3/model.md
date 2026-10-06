##### Objective Function:

$\quad \min \sum_{i \in \mathcal{M}} \sum_{j \in \mathcal{T}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \mathcal{T}} x_{ij} = 1 \quad \forall i \in \mathcal{M}$

$\sum_{i \in \mathcal{M}} x_{ij} = 1 \quad \forall j \in \mathcal{T}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \mathcal{M}, \; j \in \mathcal{T}$

---

##### Retrieved Information

{
  "machines": [
    {"id": "M1", "table_id": "file_0_view_0", "source_row": 0},
    {"id": "M2", "table_id": "file_0_view_0", "source_row": 1},
    {"id": "M3", "table_id": "file_0_view_0", "source_row": 2},
    {"id": "M4", "table_id": "file_0_view_0", "source_row": 3},
    {"id": "M5", "table_id": "file_0_view_0", "source_row": 4},
    {"id": "M6", "table_id": "file_0_view_0", "source_row": 5},
    {"id": "M7", "table_id": "file_0_view_0", "source_row": 6},
    {"id": "M8", "table_id": "file_0_view_0", "source_row": 7},
    {"id": "M9", "table_id": "file_0_view_0", "source_row": 8},
    {"id": "M10", "table_id": "file_0_view_0", "source_row": 9},
    {"id": "M11", "table_id": "file_0_view_0", "source_row": 10},
    {"id": "M12", "table_id": "file_0_view_0", "source_row": 11}
  ],
  "tasks": [
    {"id": "A", "cost_column": "assignment_cost_to_project_A"},
    {"id": "B", "cost_column": "assignment_cost_to_project_B"},
    {"id": "C", "cost_column": "assignment_cost_to_project_C"},
    {"id": "D", "cost_column": "assignment_cost_to_project_D"},
    {"id": "E", "cost_column": "assignment_cost_to_project_E"},
    {"id": "F", "cost_column": "assignment_cost_to_project_F"},
    {"id": "G", "cost_column": "assignment_cost_to_project_G"},
    {"id": "H", "cost_column": "assignment_cost_to_project_H"},
    {"id": "I", "cost_column": "assignment_cost_to_project_I"},
    {"id": "J", "cost_column": "assignment_cost_to_project_J"},
    {"id": "K", "cost_column": "assignment_cost_to_project_K"},
    {"id": "L", "cost_column": "assignment_cost_to_project_L"}
  ],
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_index": "assignee_id",
    "column_index": [
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
}

Where:
- $\mathcal{M}$ is the set of machines, indexed by "assignee_id" in table_id "file_0_view_0".
- $\mathcal{T}$ is the set of tasks, indexed by assignment cost columns as listed above.
- $c_{ij}$ is the cost of assigning machine $i$ to task $j$, given by the value in the corresponding row ("assignee_id" = $i$) and column ("assignment_cost_to_project_X" = $j$) in table_id "file_0_view_0".
- $x_{ij}$ is a binary decision variable indicating whether machine $i$ is assigned to task $j$.