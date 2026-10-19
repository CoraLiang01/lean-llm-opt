##### Decision Variables

$x_{ij}\in\{0,1\}$: assignment of manager $i\in I$ to project $j\in J$.

##### Objective Function

$\min\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ij}$

##### Constraints

1. Manager assignment: $\sum_{j\in J}x_{ij}=1,\quad\forall i\in I$  
2. Project assignment: $\sum_{i\in I}x_{ij}=1,\quad\forall j\in J$  
3. Binary: $x_{ij}\in\{0,1\}$

$I=\{MA,MB,MC\}$ and $J=\{P1,P2,P3\}$.

###### Retrieved Information

{
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
  "cost": {
    "MA": {
      "P1": 3000,
      "P2": 3200,
      "P3": 3100
    },
    "MB": {
      "P1": 2800,
      "P2": 3300,
      "P3": 2900
    },
    "MC": {
      "P1": 2900,
      "P2": 3100,
      "P3": 3000
    }
  }
}