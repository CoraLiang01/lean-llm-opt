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
    "row_index_set": "Manager",
    "column_index_set": ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"],
    "matrix": "c_{ij}"
  },
  "managers": {
    "table_id": "file_0_view_0",
    "index_set": "Manager"
  },
  "projects": {
    "table_id": "file_0_view_0",
    "index_set": ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"]
  }
}