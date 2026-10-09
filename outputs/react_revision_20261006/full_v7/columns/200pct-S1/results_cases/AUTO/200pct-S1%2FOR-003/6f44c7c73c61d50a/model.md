##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

###### 2. Variable Constraints:

$x_{mp} \in \{0,1\} \quad \forall m \in M, \; p \in P$

##### Retrieved Information

{
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_index_set": "Manager",
    "column_index_set": ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"],
    "value": "cost of assigning manager m to project p"
  },
  "managers": [
    "Manager 1",
    "Manager 2",
    "Manager 3",
    "Manager 4",
    "Manager 5",
    "Manager 6",
    "Manager 7"
  ],
  "projects": [
    "Project 1",
    "Project 2",
    "Project 3",
    "Project 4",
    "Project 5",
    "Project 6",
    "Project 7"
  ]
}