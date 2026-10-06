##### Objective Function:

$\quad \min \sum_{i \in \text{Managers}} \sum_{j \in \text{Projects}} c_{ij} \, x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \text{Projects}} x_{ij} = 1 \quad \forall i \in \text{Managers}$

$\sum_{i \in \text{Managers}} x_{ij} = 1 \quad \forall j \in \text{Projects}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Managers}, \; j \in \text{Projects}$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "manager_column": "Unnamed: 0",
    "project_columns": ["P1", "P2", "P3", "P4", "P5", "P6"]
  },
  "Managers": [
    "MA",
    "MB",
    "MC",
    "MD",
    "ME",
    "MF"
  ],
  "Projects": [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6"
  ]
}