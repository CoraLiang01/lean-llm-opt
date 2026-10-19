##### Decision Variables

$x_{ij} \ge 0$: quantity shipped from warehouse $i \in I$ to musician or band $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Branch demand: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$.  
2. Warehouse activation: $\sum_{j\in J}x_{ij}\le M y_i,\quad \forall i\in I$.  
3. Domains: $x_{ij}\ge0$ continuous; $y_i\in\{0,1\}$.

$I=\{S1,S2,S3,S4,S5,S6,S7\}$, $J=\{C1,C2,C3,C4,C5,C6,C7\}$, and $M=\sum_j d_j=37058$. There are no physical warehouse capacity limits; $M$ is a valid upper bound on total shipments.  

###### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7"
  ],
  "customers": [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7"
  ],
  "demand": {
    "C1": 1083,
    "C2": 776,
    "C3": 16214,
    "C4": 553,
    "C5": 17106,
    "C6": 594,
    "C7": 732
  },
  "fixed_cost": {
    "S1": 102.33,
    "S2": 94.92,
    "S3": 91.83,
    "S4": 98.70999999999999,
    "S5": 95.73,
    "S6": 99.95999999999999,
    "S7": 98.16
  },
  "cost": {
    "S1": {
      "C1": 1506.22,
      "C2": 70.90000000000001,
      "C3": 8.44,
      "C4": 260.27,
      "C5": 197.47,
      "C6": 71.70999999999999,
      "C7": 61.19
    },
    "S2": {
      "C1": 1732.65,
      "C2": 1780.72,
      "C3": 567.4400000000001,
      "C4": 448.68,
      "C5": 29,
      "C6": 1484.91,
      "C7": 963.92
    },
    "S3": {
      "C1": 115.66,
      "C2": 100.76,
      "C3": 64.68000000000001,
      "C4": 1324.53,
      "C5": 64.98999999999999,
      "C6": 134.88,
      "C7": 2102.83
    },
    "S4": {
      "C1": 1254.78,
      "C2": 1115.63,
      "C3": 52.31,
      "C4": 1036.16,
      "C5": 892.63,
      "C6": 1464.04,
      "C7": 1383.41
    },
    "S5": {
      "C1": 42.9,
      "C2": 891.01,
      "C3": 1013.94,
      "C4": 1128.72,
      "C5": 58.91,
      "C6": 42.89,
      "C7": 1570.31
    },
    "S6": {
      "C1": 0.7,
      "C2": 139.46,
      "C3": 70.03,
      "C4": 79.15000000000001,
      "C5": 1482,
      "C6": 0.91,
      "C7": 110.46
    },
    "S7": {
      "C1": 1732.3,
      "C2": 1780.44,
      "C3": 486.5,
      "C4": 523.74,
      "C5": 522.08,
      "C6": 82.48,
      "C7": 826.41
    }
  },
  "M": 37058
}