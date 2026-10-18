##### Decision Variables

$x_{ij} \ge 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Demand satisfaction: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$.  
2. Activation constraint: $\sum_{j\in J}x_{ij}\le M y_i,\quad \forall i\in I$.  
3. Domains: $x_{ij}\ge 0$ continuous; $y_i\in\{0,1\}$.

$I=\{S1,S2,S3\}$, $J=\{C1,C2,C3\}$, and $M=\sum_j d_j=17973$. There are no physical warehouse capacity limits; $M$ is a valid upper bound on total shipments.

###### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2",
    "S3"
  ],
  "customers": [
    "C1",
    "C2",
    "C3"
  ],
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214
  },
  "fixed_cost": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83
  },
  "cost": {
    "S1": {
      "C1": 1506.22,
      "C2": 70.9,
      "C3": 8.44
    },
    "S2": {
      "C1": 1732.65,
      "C2": 1780.72,
      "C3": 567.44
    },
    "S3": {
      "C1": 115.66,
      "C2": 100.76,
      "C3": 64.68
    }
  },
  "M": 17973
}