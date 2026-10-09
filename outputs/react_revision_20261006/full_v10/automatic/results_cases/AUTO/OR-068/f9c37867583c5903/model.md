##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

##### Retrieved Information

{
  "cost": {
    "table_id": "file_0_view_0",
    "row_ids": ["MA", "MB", "MC", "MD", "ME", "MF"],
    "col_ids": ["P1", "P2", "P3", "P4", "P5", "P6"],
    "mapping": "c_{ij} is the cost from 'manager_project_costs.csv', table_id 'file_0_view_0', row i, column j"
  },
  "managers": ["MA", "MB", "MC", "MD", "ME", "MF"],
  "projects": ["P1", "P2", "P3", "P4", "P5", "P6"]
}