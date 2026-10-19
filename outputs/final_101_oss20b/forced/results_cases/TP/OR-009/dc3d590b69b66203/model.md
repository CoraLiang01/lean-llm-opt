##### Decision Variables

$x_{ij}\geq0$: quantity shipped from plant $i\in I$ to retail outlet $j\in J$ (continuous).

##### Objective Function

$\min\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ij}$

##### Constraints

1. Demand: $\sum_{i\in I}x_{ij}\geq d_j,\quad\forall j\in J$.
2. Supply capacity: $\sum_{j\in J}x_{ij}\leq s_i,\quad\forall i\in I$.
3. Non‑negativity: $x_{ij}\geq0$ continuous.

$I=\{S1,S2,S3,S4\}$ and $J=\{C1,C2,C3,C4\}$.

###### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2",
    "S3",
    "S4"
  ],
  "customers": [
    "C1",
    "C2",
    "C3",
    "C4"
  ],
  "demand": {
    "C1": 94,
    "C2": 39,
    "C3": 65,
    "C4": 435
  },
  "supply_capacity": {
    "S1": 2531,
    "S2": 20,
    "S3": 210,
    "S4": 241
  },
  "cost": {
    "S1": {
      "C1": 543.756480860856,
      "C2": 23.685276141764653,
      "C3": 23.676386730773032,
      "C4": 447.75143678673766
    },
    "S2": {
      "C1": 883.9151090405642,
      "C2": 0.04977684765576961,
      "C3": 0.0350986687216299,
      "C4": 44.45588531711622
    },
    "S3": {
      "C1": 537.3456896658107,
      "C2": 23.769274659075112,
      "C3": 498.95659249465467,
      "C4": 440.60737890439776
    },
    "S4": {
      "C1": 1791.493192397229,
      "C2": 68.21633865655126,
      "C3": 1432.4837339656747,
      "C4": 1527.7635425462734
    }
  }
}