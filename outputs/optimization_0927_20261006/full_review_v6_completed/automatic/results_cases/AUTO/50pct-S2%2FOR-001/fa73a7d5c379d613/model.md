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
      "P1": 3000,
      "P2": 3200,
      "P3": 3100,
      "annual_training_hours": 18,
      "operations_region": "East"
    },
    "MB": {
      "P1": 2800,
      "P2": 3300,
      "P3": 2900,
      "annual_training_hours": 18,
      "operations_region": "East"
    },
    "MC": {
      "P1": 2900,
      "P2": 3100,
      "P3": 3000,
      "annual_training_hours": 12,
      "operations_region": "East"
    }
  },
  "managers": [
    {
      "id": "MA",
      "annual_training_hours": 18,
      "operations_region": "East"
    },
    {
      "id": "MB",
      "annual_training_hours": 18,
      "operations_region": "East"
    },
    {
      "id": "MC",
      "annual_training_hours": 12,
      "operations_region": "East"
    }
  ],
  "projects": [
    "P1",
    "P2",
    "P3"
  ]
}