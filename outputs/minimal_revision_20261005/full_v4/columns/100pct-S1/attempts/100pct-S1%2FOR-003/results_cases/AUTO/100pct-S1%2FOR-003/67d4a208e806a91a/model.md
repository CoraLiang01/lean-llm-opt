##### Objective Function:

$\quad \min \sum_{m \in M} \sum_{p \in P} c_{mp} \, x_{mp}$

##### Constraints

###### 1. Each manager is assigned to exactly one project:

$\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M$

###### 2. Each project is assigned to exactly one manager:

$\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P$

###### 3. Variable domains:

$x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "row_index_set": "Manager",
    "column_index_set": [
      "Project 1 Cost",
      "Project 2 Cost",
      "Project 3 Cost",
      "Project 4 Cost",
      "Project 5 Cost",
      "Project 6 Cost",
      "Project 7 Cost"
    ],
    "mapping": "c_{mp} = \u2018Project k Cost\u2019 value for manager m in row 'Manager', column 'Project k Cost', in table_id file_0_view_0"
  },
  "managers": {
    "index_set": [
      "Manager 1",
      "Manager 2",
      "Manager 3",
      "Manager 4",
      "Manager 5",
      "Manager 6",
      "Manager 7"
    ],
    "table_id": "file_0_view_0",
    "column": "Manager"
  },
  "projects": {
    "index_set": [
      "Project 1 Cost",
      "Project 2 Cost",
      "Project 3 Cost",
      "Project 4 Cost",
      "Project 5 Cost",
      "Project 6 Cost",
      "Project 7 Cost"
    ],
    "table_id": "file_0_view_0",
    "column": [
      "Project 1 Cost",
      "Project 2 Cost",
      "Project 3 Cost",
      "Project 4 Cost",
      "Project 5 Cost",
      "Project 6 Cost",
      "Project 7 Cost"
    ]
  }
}