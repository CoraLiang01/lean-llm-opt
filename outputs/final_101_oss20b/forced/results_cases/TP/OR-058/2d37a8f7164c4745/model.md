##### Decision Variables

$y_{i}\in\{0,1\}$: binary variable indicating whether supplier $i\in I$ is operational (open).  
$x_{ij}\geq0$: continuous variable representing the quantity of goods shipped from supplier $i\in I$ to customer $j\in J$.

##### Objective Function

$\displaystyle \min\Bigl(\sum_{i\in I}f_{i}y_{i}\;+\;\sum_{i\in I}\sum_{j\in J}c_{ij}x_{ij}\Bigr)$

##### Constraints

1. Demand satisfaction:  
   $\displaystyle \sum_{i\in I}x_{ij}\;\geq\;d_{j},\quad\forall j\in J$  

2. Supplier activation linkage (big‑M):  
   $\displaystyle x_{ij}\;\leq\;M\,y_{i},\quad\forall i\in I,\;\forall j\in J$  

   where $M=1080$ (the total demand across all customers).  

3. Non‑negativity:  
   $\displaystyle x_{ij}\;\geq\;0,\quad\forall i\in I,\;\forall j\in J$  

4. Binary requirement:  
   $\displaystyle y_{i}\in\{0,1\},\quad\forall i\in I$  

$ I=\{S1,S2,S3,S4,S5,S6\}$ and $ J=\{C1,C2,C3,C4,C5,C6\}$.

###### Retrieved Information

{
  "suppliers": [
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6"
  ],
  "customers": [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6"
  ],
  "demand": {
    "C1": 216,
    "C2": 216,
    "C3": 216,
    "C4": 144,
    "C5": 144,
    "C6": 144
  },
  "fixed_cost": {
    "S1": 98.88,
    "S2": 99.73,
    "S3": 94.01,
    "S4": 93.77,
    "S5": 107.59,
    "S6": 112.65
  },
  "cost": {
    "S1": {
      "C1": 0.08,
      "C2": 52.33,
      "C3": 73.57,
      "C4": 1237.33,
      "C5": 0.07,
      "C6": 112.16
    },
    "S2": {
      "C1": 46.02,
      "C2": 175.23,
      "C3": 2026.83,
      "C4": 299.89,
      "C5": 966.53,
      "C6": 1590.42
    },
    "S3": {
      "C1": 1031.74,
      "C2": 78.13,
      "C3": 99.02,
      "C4": 277.07,
      "C5": 884.45,
      "C6": 1800.86
    },
    "S4": {
      "C1": 868.75,
      "C2": 94.20,
      "C3": 1776.34,
      "C4": 285.48,
      "C5": 868.85,
      "C6": 86.55
    },
    "S5": {
      "C1": 1577.00,
      "C2": 760.15,
      "C3": 2090.19,
      "C4": 43.20,
      "C5": 1577.12,
      "C6": 1095.17
    },
    "S6": {
      "C1": 49.14,
      "C2": 4.33,
      "C3": 2079.57,
      "C4": 277.04,
      "C5": 1032.01,
      "C6": 1543.49
    }
  }
}