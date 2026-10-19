##### Decision Variables

$y_i\in\{0,1\}$: binary variable indicating whether warehouse $i\in I$ is operational.  
$x_{ij}\geq0$: quantity of goods shipped from warehouse $i\in I$ to customer $j\in J$ (continuous).

##### Objective Function

$\displaystyle \min\Bigl(\sum_{i\in I}f_i\,y_i+\sum_{i\in I}\sum_{j\in J}c_{ij}\,x_{ij}\Bigr)$

##### Constraints

1. Demand satisfaction (equality):  
   $\displaystyle \sum_{i\in I}x_{ij}=d_j,\quad\forall j\in J$.

2. Warehouse activation linkage:  
   $\displaystyle x_{ij}\leq d_j\,y_i,\quad\forall i\in I,\;\forall j\in J$.

3. Non‑negativity:  
   $\displaystyle x_{ij}\geq0,\quad\forall i\in I,\;\forall j\in J$.

4. Binary requirement:  
   $\displaystyle y_i\in\{0,1\},\quad\forall i\in I$.

$ I=\{S1,S2,S3\}$ and $J=\{C1,C2,C3\}$.

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
  "fixed_costs": {
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
  }
}