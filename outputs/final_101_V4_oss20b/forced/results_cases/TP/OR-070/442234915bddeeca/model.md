##### Decision Variables

$x_{ij}\in\{0,1\}$: 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Objective Function

$\min\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ij}$

##### Constraints

1. Manager assignment: $\sum_{j\in J}x_{ij}=1,\quad\forall i\in I$.
2. Project assignment: $\sum_{i\in I}x_{ij}=1,\quad\forall j\in J$.
3. Binary: $x_{ij}\in\{0,1\},\quad\forall i\in I,\forall j\in J$.

$I=\{\text{Manager 1},\text{Manager 2},\text{Manager 3},\text{Manager 4},\text{Manager 5},\text{Manager 6},\text{Manager 7}\}$

$J=\{\text{Project 1},\text{Project 2},\text{Project 3},\text{Project 4},\text{Project 5},\text{Project 6},\text{Project 7}\}$

###### Retrieved Information

{
  "managers": [
    "Manager 1",
    "Manager 2",
    "Manager 3",
    "Manager 4",
    "Manager 5",
    "Manager 6",
    "Manager 7"
  ],
  "projects": [
    "Project 1",
    "Project 2",
    "Project 3",
    "Project 4",
    "Project 5",
    "Project 6",
    "Project 7"
  ],
  "cost": {
    "Manager 1": {
      "Project 1": 2972,
      "Project 2": 2727,
      "Project 3": 2795,
      "Project 4": 2922,
      "Project 5": 1302,
      "Project 6": 2489,
      "Project 7": 1533
    },
    "Manager 2": {
      "Project 1": 1094,
      "Project 2": 2158,
      "Project 3": 2990,
      "Project 4": 1844,
      "Project 5": 2887,
      "Project 6": 2021,
      "Project 7": 2288
    },
    "Manager 3": {
      "Project 1": 2133,
      "Project 2": 1675,
      "Project 3": 2422,
      "Project 4": 2639,
      "Project 5": 1033,
      "Project 6": 2261,
      "Project 7": 1695
    },
    "Manager 4": {
      "Project 1": 1951,
      "Project 2": 2309,
      "Project 3": 2070,
      "Project 4": 2802,
      "Project 5": 2328,
      "Project 6": 1313,
      "Project 7": 2434
    },
    "Manager 5": {
      "Project 1": 1269,
      "Project 2": 2153,
      "Project 3": 1296,
      "Project 4": 2685,
      "Project 5": 2627,
      "Project 6": 1610,
      "Project 7": 1641
    },
    "Manager 6": {
      "Project 1": 1220,
      "Project 2": 1192,
      "Project 3": 2907,
      "Project 4": 2622,
      "Project 5": 2595,
      "Project 6": 1261,
      "Project 7": 2384
    },
    "Manager 7": {
      "Project 1": 1286,
      "Project 2": 1659,
      "Project 3": 1179,
      "Project 4": 1348,
      "Project 5": 1420,
      "Project 6": 2862,
      "Project 7": 1959
    }
  }
}