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
  "cost": {
    "table_id": "file_0_view_0",
    "row_index_set": ["Manager 1", "Manager 2", "Manager 3", "Manager 4", "Manager 5", "Manager 6", "Manager 7"],
    "column_index_set": ["Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"],
    "value_mapping": "c_{ij} = value in 'Project k Cost' column for manager i in 'Manager' row"
  },
  "Managers": [
    "Manager 1",
    "Manager 2",
    "Manager 3",
    "Manager 4",
    "Manager 5",
    "Manager 6",
    "Manager 7"
  ],
  "Projects": [
    "Project 1 Cost",
    "Project 2 Cost",
    "Project 3 Cost",
    "Project 4 Cost",
    "Project 5 Cost",
    "Project 6 Cost",
    "Project 7 Cost"
  ]
}