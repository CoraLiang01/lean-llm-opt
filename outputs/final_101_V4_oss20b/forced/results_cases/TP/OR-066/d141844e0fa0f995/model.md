##### Decision Variables

$x_{ij}\geq0$: quantity shipped from supplier $i\in I$ to supermarket $j\in J$ (continuous).  
$y_{i}\in\{0,1\}$: binary variable indicating whether supplier $i$ is activated.

##### Objective Function

$\displaystyle \min\Bigl(\sum_{i\in I}f_{i}y_{i}+\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ij}\Bigr)$

##### Constraints

1. Demand satisfaction: $\displaystyle \sum_{i\in I}x_{ij}\geq d_{j},\quad\forall j\in J$  
2. Linking constraint: $\displaystyle x_{ij}\leq d_{j}\,y_{i},\quad\forall i\in I,\;\forall j\in J$  
3. Non‑negativity: $\displaystyle x_{ij}\geq0,\quad\forall i\in I,\;\forall j\in J$  
4. Binary: $\displaystyle y_{i}\in\{0,1\},\quad\forall i\in I$

$I=\{S1,S2\}$ and $J=\{C1,C2\}$.

###### Retrieved Information

```json
{
  "suppliers": [
    "S1",
    "S2"
  ],
  "customers": [
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
  "cost": {
    "S1": {
      "C1": 2358.39,
      "C2": 1492.08
    },
    "S2": {
      "C1": 0.07,
      "C2": 52.32
    }
  }
}
```