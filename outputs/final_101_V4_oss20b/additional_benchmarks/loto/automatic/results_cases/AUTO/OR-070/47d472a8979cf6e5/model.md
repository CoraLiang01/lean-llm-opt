##### Decision Variables

$x_{ij}\in\{0,1\}$: assignment of manager $i\in I$ to project $j\in J$.

##### Objective Function

$\min\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ij}$

##### Constraints

1. Project assignment: $\sum_{i\in I}x_{ij}=1,\quad\forall j\in J$.
2. Manager assignment: $\sum_{j\in J}x_{ij}=1,\quad\forall i\in I$.
3. Binary: $x_{ij}\in\{0,1\}$.

$I=\{M1,M2,M3,M4,M5,M6,M7\}$ and $J=\{P1,P2,P3,P4,P5,P6,P7\}$.

###### Retrieved Information

{
  "managers": [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7"
  ],
  "projects": [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7"
  ],
  "cost": {
    "M1": {
      "P1": 2972,
      "P2": 2727,
      "P3": 2795,
      "P4": 2922,
      "P5": 1302,
      "P6": 2489,
      "P7": 1533
    },
    "M2": {
      "P1": 1094,
      "P2": 2158,
      "P3": 2990,
      "P4": 1844,
      "P5": 2887,
      "P6": 2021,
      "P7": 2288
    },
    "M3": {
      "P1": 2133,
      "P2": 1675,
      "P3": 2422,
      "P4": 2639,
      "P5": 1033,
      "P6": 2261,
      "P7": 1695
    },
    "M4": {
      "P1": 1951,
      "P2": 2309,
      "P3": 2070,
      "P4": 2802,
      "P5": 2328,
      "P6": 1313,
      "P7": 2434
    },
    "M5": {
      "P1": 1269,
      "P2": 2153,
      "P3": 1296,
      "P4": 2685,
      "P5": 2627,
      "P6": 1610,
      "P7": 1641
    },
    "M6": {
      "P1": 1220,
      "P2": 1192,
      "P3": 2907,
      "P4": 2622,
      "P5": 2595,
      "P6": 1261,
      "P7": 2384
    },
    "M7": {
      "P1": 1286,
      "P2": 1659,
      "P3": 1179,
      "P4": 1348,
      "P5": 1420,
      "P6": 2862,
      "P7": 1959
    }
  }
}