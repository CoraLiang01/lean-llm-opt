##### Objective Function:

$\quad \min \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j=1}^{11} x_{ij} = 1 \quad \forall i \in \{1,2,\ldots,11\}$

$\sum_{i=1}^{11} x_{ij} = 1 \quad \forall j \in \{1,2,\ldots,11\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i,j$

##### Parameter and Data Mapping

- $c_{ij}$: Cost of assigning manager $i$ to project $j$, from table_id: file_0_view_0, with row_id_mapping: "Manager" and column_id_mapping: "Project 1 Cost", "Project 2 Cost", ..., "Project 11 Cost".
- $x_{ij}$: Binary variable, equals 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Data Mapping

{
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_id_mapping": "Manager",
    "column_id_mapping": [
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
    ]
  },
  "managers": {
    "table_id": "file_0_view_0",
    "row_id_mapping": "Manager"
  },
  "projects": {
    "table_id": "file_0_view_0",
    "column_id_mapping": [
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
    ]
  }
}