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
      "MA": {"P1": "2216", "P2": "1911", "P3": "1661", "P4": "2122", "P5": "1442", "P6": "1442"},
      "MB": {"P1": "1100", "P2": "1271", "P3": "2764", "P4": "2557", "P5": "1036", "P6": "1036"},
      "MC": {"P1": "2827", "P2": "2784", "P3": "2206", "P4": "2216", "P5": "2677", "P6": "2677"},
      "MD": {"P1": "2627", "P2": "1273", "P3": "2610", "P4": "1957", "P5": "1594", "P6": "1594"},
      "ME": {"P1": "3359", "P2": "1003", "P3": "2554", "P4": "1706", "P5": "2065", "P6": "2065"},
      "MF": {"P1": "1579", "P2": "2289", "P3": "2368", "P4": "1922", "P5": "2740", "P6": "2740"}
    }
  },
  "managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}