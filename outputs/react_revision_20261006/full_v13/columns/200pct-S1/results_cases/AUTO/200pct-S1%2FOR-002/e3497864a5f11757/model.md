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
    "manager_ids": ["MA", "MB", "MC", "MD", "ME", "MF"],
    "project_ids": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "cost_column_mapping": {
      "MA": {"P1": "file_0_view_0:MA:P1", "P2": "file_0_view_0:MA:P2", "P3": "file_0_view_0:MA:P3", "P4": "file_0_view_0:MA:P4", "P5": "file_0_view_0:MA:P5", "P6": "file_0_view_0:MA:P6"},
      "MB": {"P1": "file_0_view_0:MB:P1", "P2": "file_0_view_0:MB:P2", "P3": "file_0_view_0:MB:P3", "P4": "file_0_view_0:MB:P4", "P5": "file_0_view_0:MB:P5", "P6": "file_0_view_0:MB:P6"},
      "MC": {"P1": "file_0_view_0:MC:P1", "P2": "file_0_view_0:MC:P2", "P3": "file_0_view_0:MC:P3", "P4": "file_0_view_0:MC:P4", "P5": "file_0_view_0:MC:P5", "P6": "file_0_view_0:MC:P6"},
      "MD": {"P1": "file_0_view_0:MD:P1", "P2": "file_0_view_0:MD:P2", "P3": "file_0_view_0:MD:P3", "P4": "file_0_view_0:MD:P4", "P5": "file_0_view_0:MD:P5", "P6": "file_0_view_0:MD:P6"},
      "ME": {"P1": "file_0_view_0:ME:P1", "P2": "file_0_view_0:ME:P2", "P3": "file_0_view_0:ME:P3", "P4": "file_0_view_0:ME:P4", "P5": "file_0_view_0:ME:P5", "P6": "file_0_view_0:ME:P6"},
      "MF": {"P1": "file_0_view_0:MF:P1", "P2": "file_0_view_0:MF:P2", "P3": "file_0_view_0:MF:P3", "P4": "file_0_view_0:MF:P4", "P5": "file_0_view_0:MF:P5", "P6": "file_0_view_0:MF:P6"}
    }
  },
  "managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}