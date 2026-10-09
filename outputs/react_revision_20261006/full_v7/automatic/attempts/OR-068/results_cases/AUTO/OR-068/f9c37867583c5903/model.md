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
    "row_index_set": ["MA", "MB", "MC", "MD", "ME", "MF"],
    "column_index_set": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "value": "c_{ij} = \text{cost of assigning manager } i \text{ to project } j \text{ from table file_0_view_0, row } i, \text{ column } j"
  },
  "Managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "Projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}