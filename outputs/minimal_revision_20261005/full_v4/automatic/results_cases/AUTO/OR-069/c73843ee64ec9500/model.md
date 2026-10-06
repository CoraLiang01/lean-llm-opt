##### Objective Function:

$\quad \min \sum_{i \in M} \sum_{j \in P} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M$

$\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in M, \; j \in P$

##### Retrieved Information

{
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_index_set": "Manager",
    "column_index_set": [
      "Project 1 Cost",
      "Project 2 Cost",
      "Project 3 Cost",
      "Project 4 Cost",
      "Project 5 Cost",
      "Project 6 Cost",
      "Project 7 Cost",
      "Project 8 Cost",
      "Project 9 Cost",
      "Project 10 Cost",
      "Project 11 Cost"
    ],
    "row_entities": [
      "Manager 1",
      "Manager 2",
      "Manager 3",
      "Manager 4",
      "Manager 5",
      "Manager 6",
      "Manager 7",
      "Manager 8",
      "Manager 9",
      "Manager 10",
      "Manager 11"
    ]
  },
  "managers": {
    "set": "M",
    "elements": [
      "Manager 1",
      "Manager 2",
      "Manager 3",
      "Manager 4",
      "Manager 5",
      "Manager 6",
      "Manager 7",
      "Manager 8",
      "Manager 9",
      "Manager 10",
      "Manager 11"
    ]
  },
  "projects": {
    "set": "P",
    "elements": [
      "Project 1",
      "Project 2",
      "Project 3",
      "Project 4",
      "Project 5",
      "Project 6",
      "Project 7",
      "Project 8",
      "Project 9",
      "Project 10",
      "Project 11"
    ]
  }
}