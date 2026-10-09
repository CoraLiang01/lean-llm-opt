##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "manager_index": "Unnamed: 0",
    "project_indices": ["P1", "P2", "P3", "P4", "P5", "P6"]
  },
  "managers": [
    {"id": "MA", "table_id": "file_0_view_0", "source_row": 0},
    {"id": "MB", "table_id": "file_0_view_0", "source_row": 1},
    {"id": "MC", "table_id": "file_0_view_0", "source_row": 2},
    {"id": "MD", "table_id": "file_0_view_0", "source_row": 3},
    {"id": "ME", "table_id": "file_0_view_0", "source_row": 4},
    {"id": "MF", "table_id": "file_0_view_0", "source_row": 5}
  ],
  "projects": [
    "P1", "P2", "P3", "P4", "P5", "P6"
  ]
}