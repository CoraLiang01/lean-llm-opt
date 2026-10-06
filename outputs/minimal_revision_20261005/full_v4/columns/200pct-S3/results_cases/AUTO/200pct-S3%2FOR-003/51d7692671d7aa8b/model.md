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
    "row_ids": [
      "Manager 1",
      "Manager 2",
      "Manager 3",
      "Manager 4",
      "Manager 5",
      "Manager 6",
      "Manager 7"
    ]
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

- $M$: set of managers, from "Manager" column in table_id file_0_view_0
- $P$: set of projects, corresponding to columns "Project 1 Cost", ..., "Project 7 Cost" in table_id file_0_view_0
- $c_{ij}$: cost for manager $i$ to complete project $j$, from table_id file_0_view_0, row $i$, column $j$ ("Project k Cost")
- $x_{ij}$: binary variable, 1 if manager $i$ is assigned to project $j$, 0 otherwise

**Data Mapping:** All cost parameters $c_{ij}$ are mapped to table_id file_0_view_0, with $i$ indexed by "Manager" and $j$ by "Project k Cost" columns. All managers and projects are included as listed above.