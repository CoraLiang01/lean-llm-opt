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
    "cost_matrix": "c_{ij} = value in column Pj for row manager i in table_id file_0_view_0"
  },
  "managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}