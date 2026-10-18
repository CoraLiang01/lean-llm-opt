##### Decision Variables

$x_{ij}\geq0$: quantity shipped from supplier $i\in I$ to customer $j\in J$ (continuous).  
$y_i\in\{0,1\}$: supplier $i$ is operational (open).

##### Objective Function

$\min \sum_{i\in I} f_i y_i + \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij}$

##### Constraints

1. Demand satisfaction:  
   $\displaystyle \sum_{i\in I} x_{ij} \ge d_j,\quad \forall j\in J$.

2. Linking (no shipment if supplier closed):  
   $\displaystyle x_{ij} \le M\,y_i,\quad \forall i\in I,\ \forall j\in J$  
   (where $M$ is a sufficiently large constant, e.g., the sum of all demands).

3. Non‑negativity:  
   $\displaystyle x_{ij}\geq0,\quad \forall i\in I,\ \forall j\in J$.

4. Binary requirement:  
   $\displaystyle y_i\in\{0,1\},\quad \forall i\in I$.

$I=\{S1,S2,S3,S4,S5\}$ and $J=\{C1,C2,C3,C4,C5\}$.

###### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5"
  ],
  "customers": [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5"
  ],
  "demand": {
    "C1": 143,
    "C2": 6,
    "C3": 10,
    "C4": 25,
    "C5": 3
  },
  "fixed_costs": {
    "S1": 97.65000000000001,
    "S2": 99.76000000000001,
    "S3": 100.76,
    "S4": 105.32,
    "S5": 98.88
  },
  "cost": {
    "S1": {
      "C1": 150.74,
      "C2": 0.02,
      "C3": 49.13,
      "C4": 2080.15,
      "C5": 426.4
    },
    "S2": {
      "C1": 233.05,
      "C2": 97.73,
      "C3": 49.84,
      "C4": 1982.39,
      "C5": 23.96
    },
    "S3": {
      "C1": 55.68,
      "C2": 935.61,
      "C3": 4.03,
      "C4": 73.09,
      "C5": 525.3200000000001
    },
    "S4": {
      "C1": 1483.82,
      "C2": 1801.08,
      "C3": 112.16,
      "C4": 816.05,
      "C5": 107.01
    },
    "S5": {
      "C1": 1119.47,
      "C2": 884.3099999999999,
      "C3": 0.08,
      "C4": 1544.95,
      "C5": 543.67
    }
  }
}