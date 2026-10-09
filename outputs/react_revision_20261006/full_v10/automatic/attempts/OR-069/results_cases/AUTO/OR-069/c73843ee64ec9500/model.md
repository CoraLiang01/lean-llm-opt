##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers},\ \forall j \in \text{Projects}$

##### Retrieved Information

{
  "cost_matrix": {
    "table_id": "file_0_view_0",
    "row_ids": [
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
    ],
    "column_ids": [
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
  "Managers": [
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
  ],
  "Projects": [
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