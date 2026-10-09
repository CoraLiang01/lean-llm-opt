##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

###### 2. Variable Constraints:

$x_{mp} \in \{0,1\} \quad \forall m \in M, \; p \in P$

###### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "row_id_column": "Unnamed: 1",
    "column_ids": ["P1", "P2", "P3"],
    "row_ids": ["MA", "MB", "MC"],
    "value_column_matrix": {
      "MA": {"P1": "3000", "P2": "3200", "P3": "3100"},
      "MB": {"P1": "2800", "P2": "3300", "P3": "2900"},
      "MC": {"P1": "2900", "P2": "3100", "P3": "3000"}
    }
  },
  "managers": ["MA", "MB", "MC"],
  "projects": ["P1", "P2", "P3"]
}