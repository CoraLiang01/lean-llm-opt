##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

###### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "manager_ids": ["MA", "MB", "MC", "MD", "ME", "MF"],
    "project_ids": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "cost_column_names": ["P1", "P2", "P3", "P4", "P5", "P6"]
  }
}

Where:
- $M$ is the set of managers: all "Unnamed: 0" values in table_id "file_0_view_0"
- $P$ is the set of projects: all columns ["P1", "P2", "P3", "P4", "P5", "P6"] in table_id "file_0_view_0"
- $c_{ij}$ is the assignment cost for manager $i$ to project $j$, from the corresponding cell in table_id "file_0_view_0"
- $x_{ij}$ is a binary variable equal to 1 if manager $i$ is assigned to project $j$, 0 otherwise