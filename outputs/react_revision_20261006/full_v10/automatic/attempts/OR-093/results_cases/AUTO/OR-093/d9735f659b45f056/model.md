##### Objective Function:

$\quad \min \sum_{i \in \text{Machines}} \sum_{j \in \text{Tasks}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \text{Tasks}} x_{ij} = 1 \quad \forall i \in \text{Machines}$

$\sum_{i \in \text{Machines}} x_{ij} = 1 \quad \forall j \in \text{Tasks}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \text{Machines},\ \forall j \in \text{Tasks}$

###### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "row_ids": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"],
    "column_ids": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
  },
  "Machines": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M12"],
  "Tasks": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
}