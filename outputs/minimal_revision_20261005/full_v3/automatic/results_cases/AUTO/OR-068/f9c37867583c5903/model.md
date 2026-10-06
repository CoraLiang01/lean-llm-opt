##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 3. Variable domains:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \; j \in P$

##### Retrieved Information

{
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_id_mapping": {
      "MA": 0,
      "MB": 1,
      "MC": 2,
      "MD": 3,
      "ME": 4,
      "MF": 5
    },
    "column_id_mapping": {
      "P1": 0,
      "P2": 1,
      "P3": 2,
      "P4": 3,
      "P5": 4,
      "P6": 5
    },
    "value_column_names": ["P1", "P2", "P3", "P4", "P5", "P6"]
  },
  "managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}