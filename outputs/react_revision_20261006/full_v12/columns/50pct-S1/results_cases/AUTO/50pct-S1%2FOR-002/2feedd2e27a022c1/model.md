##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}$

##### Constraints

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

$x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "manager_ids": ["MA", "MB", "MC", "MD", "ME", "MF"],
    "project_ids": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "cost_matrix_columns": ["P1", "P2", "P3", "P4", "P5", "P6"]
  }
}

Where:
- $M$ is the set of managers: ["MA", "MB", "MC", "MD", "ME", "MF"]
- $P$ is the set of projects: ["P1", "P2", "P3", "P4", "P5", "P6"]
- $c_{ij}$ is the cost of assigning manager $i$ to project $j$, from "manager_project_costs.csv", table_id "file_0_view_0", columns ["P1", "P2", "P3", "P4", "P5", "P6"]
- $x_{ij}$ is a binary variable: $x_{ij} = 1$ if manager $i$ is assigned to project $j$, $0$ otherwise.