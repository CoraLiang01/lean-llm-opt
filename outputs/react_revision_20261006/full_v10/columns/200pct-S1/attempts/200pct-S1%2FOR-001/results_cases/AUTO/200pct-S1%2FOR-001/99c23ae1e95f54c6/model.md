##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} c_{ij} x_{ij}$

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\}$

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}$

##### Retrieved Information

{
  "cost": {
    "MA": {
      "P1": "file_0_view_0: row 'MA', column 'P1'",
      "P2": "file_0_view_0: row 'MA', column 'P2'",
      "P3": "file_0_view_0: row 'MA', column 'P3'"
    },
    "MB": {
      "P1": "file_0_view_0: row 'MB', column 'P1'",
      "P2": "file_0_view_0: row 'MB', column 'P2'",
      "P3": "file_0_view_0: row 'MB', column 'P3'"
    },
    "MC": {
      "P1": "file_0_view_0: row 'MC', column 'P1'",
      "P2": "file_0_view_0: row 'MC', column 'P2'",
      "P3": "file_0_view_0: row 'MC', column 'P3'"
    }
  },
  "managers": [
    "MA",
    "MB",
    "MC"
  ],
  "projects": [
    "P1",
    "P2",
    "P3"
  ],
  "table_id": "file_0_view_0"
}