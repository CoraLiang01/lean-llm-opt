##### Objective Function:

$\quad \min \sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} \sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} c_{ij} x_{ij}$

where $c_{ij}$ is the cost of assigning manager $i$ to project $j$, as given below.

##### Constraints

###### 1. Assignment Constraints:

$\sum_{j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}} x_{ij} = 1 \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}$

$\sum_{i \in \{\text{MA}, \text{MB}, \text{MC}, \text{MD}, \text{ME}, \text{MF}\}} x_{ij} = 1 \quad \forall j \in \{\text{P1}, \text{P2}, \text{P3}, \text{P4}, \text{P5}, \text{P6}\}$

###### 2. Variable Constraints:

$x_{ij} \in \{0,1\} \quad \forall i, j$

##### Retrieved Information

{
  "cost": {
    "MA": {
      "P1": 2216,
      "P2": 1911,
      "P3": 1661,
      "P4": 2122,
      "P5": 1442,
      "P6": 1442
    },
    "MB": {
      "P1": 1100,
      "P2": 1271,
      "P3": 2764,
      "P4": 2557,
      "P5": 1036,
      "P6": 1036
    },
    "MC": {
      "P1": 2827,
      "P2": 2784,
      "P3": 2206,
      "P4": 2216,
      "P5": 2677,
      "P6": 2677
    },
    "MD": {
      "P1": 2627,
      "P2": 1273,
      "P3": 2610,
      "P4": 1957,
      "P5": 1594,
      "P6": 1594
    },
    "ME": {
      "P1": 3359,
      "P2": 1003,
      "P3": 2554,
      "P4": 1706,
      "P5": 2065,
      "P6": 2065
    },
    "MF": {
      "P1": 1579,
      "P2": 2289,
      "P3": 2368,
      "P4": 1922,
      "P5": 2740,
      "P6": 2740
    }
  },
  "managers": [
    "MA",
    "MB",
    "MC",
    "MD",
    "ME",
    "MF"
  ],
  "projects": [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6"
  ]
}