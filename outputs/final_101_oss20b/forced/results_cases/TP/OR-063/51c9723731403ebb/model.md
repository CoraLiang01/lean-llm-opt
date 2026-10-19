##### Decision Variables

$y_i\in\{0,1\}$: binary variable indicating whether warehouse $i\in I$ is operational.  
$x_{ij}\geq0$: quantity of goods shipped from warehouse $i\in I$ to customer $j\in J$ (continuous).

##### Objective Function

$\displaystyle \min\;\sum_{i\in I}\bigl(f_i\,y_i\bigr)+\sum_{i\in I}\sum_{j\in J}c_{ij}\,x_{ij}$

##### Constraints

1. Demand satisfaction:  
   $\displaystyle \sum_{i\in I}x_{ij}\;\geq\;d_j,\quad\forall j\in J$.

2. Linking shipments to activation (big‑$M$):  
   $\displaystyle x_{ij}\;\leq\;M\,y_i,\quad\forall i\in I,\;\forall j\in J$,  
   where $M=37058$ (sum of all customer demands).

3. Non‑negativity:  
   $\displaystyle x_{ij}\geq0,\quad\forall i\in I,\;\forall j\in J$.

4. Binary:  
   $\displaystyle y_i\in\{0,1\},\quad\forall i\in I$.

$I=\{S1,S2,S3,S4,S5,S6,S7\}$, $J=\{C1,C2,C3,C4,C5,C6,C7\}$.

###### Retrieved Information

{
  "suppliers": ["S1","S2","S3","S4","S5","S6","S7"],
  "customers": ["C1","C2","C3","C4","C5","C6","C7"],
  "demand": {
    "C1":1083,
    "C2":776,
    "C3":16214,
    "C4":553,
    "C5":17106,
    "C6":594,
    "C7":732
  },
  "fixed_cost": {
    "S1":102.33,
    "S2":94.92,
    "S3":91.83,
    "S4":98.71,
    "S5":95.73,
    "S6":99.96,
    "S7":98.16
  },
  "cost": {
    "S1": {
      "C1":1506.22,
      "C2":70.9,
      "C3":8.44,
      "C4":260.27,
      "C5":197.47,
      "C6":71.71,
      "C7":61.19
    },
    "S2": {
      "C1":1732.65,
      "C2":1780.72,
      "C3":567.44,
      "C4":448.68,
      "C5":29,
      "C6":1484.91,
      "C7":963.92
    },
    "S3": {
      "C1":115.66,
      "C2":100.76,
      "C3":64.68,
      "C4":1324.53,
      "C5":64.99,
      "C6":134.88,
      "C7":2102.83
    },
    "S4": {
      "C1":1254.78,
      "C2":1115.63,
      "C3":52.31,
      "C4":1036.16,
      "C5":892.63,
      "C6":1464.04,
      "C7":1383.41
    },
    "S5": {
      "C1":42.9,
      "C2":891.01,
      "C3":1013.94,
      "C4":1128.72,
      "C5":58.91,
      "C6":42.89,
      "C7":1570.31
    },
    "S6": {
      "C1":0.7,
      "C2":139.46,
      "C3":70.03,
      "C4":79.15,
      "C5":1482,
      "C6":0.91,
      "C7":110.46
    },
    "S7": {
      "C1":1732.3,
      "C2":1780.44,
      "C3":486.5,
      "C4":523.74,
      "C5":522.08,
      "C6":82.48,
      "C7":826.41
    }
  }
}