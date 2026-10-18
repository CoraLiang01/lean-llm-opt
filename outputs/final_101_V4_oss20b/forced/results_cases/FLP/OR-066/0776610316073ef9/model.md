##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Objective Function

$\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij}+\sum_{i\in I}f_i y_i$

##### Constraints

1. Supermarket demand: $\sum_{i\in I}x_{ij}=d_j,\quad \forall j\in J$.  
2. Supplier activation: $\sum_{j\in J}x_{ij}\leq M y_i,\quad \forall i\in I$.  
3. Domains: $x_{ij}\geq0$ continuous; $y_i\in\{0,1\}$.

$I=\{S1,S2\}$, $J=\{C1,C2\}$, and $M=\sum_j d_j=360$. There are no physical supplier capacity limits; $M$ is a valid upper bound on total shipments. CSV matrix rows are suppliers and columns are supermarkets.

###### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2"
  ],
  "supermarkets": [
    "C1",
    "C2"
  ],
  "demand": {
    "C1": 144,
    "C2": 216
  },
  "fixed_cost": {
    "S1": 105.97,
    "S2": 85.31
  },
  "transportation_costs": {
    "S1": {
      "C1": 2358.39,
      "C2": 1492.08
    },
    "S2": {
      "C1": 0.07,
      "C2": 52.32
    }
  },
  "M": 360
}