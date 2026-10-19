##### Objective Function: 

$\quad \quad \min \sum_{i} \sum_{j} c_{ij} \cdot x_{ij}$

##### Constraints

###### 1. Manager Assignment Constraints: 

$\quad \quad \sum_{j} x_{ij} = 1, \quad \forall i$

###### 2. Project Assignment Constraints: 

$\quad \quad \sum_{i} x_{ij} = 1, \quad \forall j$

###### 3. Binary Decision Variables: 

$\quad \quad x_{ij} \in \{0,1\}, \quad \forall i,j$

The indices $i$ run over the managers MA, MB, MC, MD, ME, MF and $j$ run over the projects P1, P2, P3, P4, P5, P6. The parameter $c_{ij}$ is the cost of assigning manager $i$ to project $j$ as given in the data below.

###### Retrieved Information

{
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
  ],
  "costs": {
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
  }
}