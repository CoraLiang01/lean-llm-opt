##### Sets

- Warehouses: $I = \{S1, S2, S3, S4, S5, S6, S7\}$
- Musicians/Bands: $J = \{C1, C2, C3, C4, C5, C6, C7\}$

##### Parameters

- Demand for each musician/band:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$
  - $d_{C4} = 553$
  - $d_{C5} = 17106$
  - $d_{C6} = 594$
  - $d_{C7} = 732$

- Fixed cost for each warehouse:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$
  - $f_{S4} = 98.71$
  - $f_{S5} = 95.73$
  - $f_{S6} = 99.96$
  - $f_{S7} = 98.16$

- Transportation cost per unit from warehouse $i$ to musician/band $j$ ($c_{ij}$):

|        | C1      | C2      | C3      | C4      | C5      | C6      | C7      |
|--------|---------|---------|---------|---------|---------|---------|---------|
| S1     | 1506.22 | 70.90   | 8.44    | 260.27  | 197.47  | 71.71   | 61.19   |
| S2     | 1732.65 | 1780.72 | 567.44  | 448.68  | 29.00   | 1484.91 | 963.92  |
| S3     | 115.66  | 100.76  | 64.68   | 1324.53 | 64.99   | 134.88  | 2102.83 |
| S4     | 1254.78 | 1115.63 | 52.31   | 1036.16 | 892.63  | 1464.04 | 1383.41 |
| S5     | 42.90   | 891.01  | 1013.94 | 1128.72 | 58.91   | 42.89   | 1570.31 |
| S6     | 0.70    | 139.46  | 70.03   | 79.15   | 1482.00 | 0.91    | 110.46  |
| S7     | 1732.30 | 1780.44 | 486.50  | 523.74  | 522.08  | 82.48   | 826.41  |

##### Decision Variables

- $y_i \in \{0,1\}$: 1 if warehouse $i$ is operational, 0 otherwise, for all $i \in I$
- $x_{ij} \geq 0$: quantity supplied from warehouse $i$ to musician/band $j$, for all $i \in I$, $j \in J$

##### Objective Function

Minimize total cost (fixed + transportation):

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction for each musician/band:
   $$
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   $$

2. Linking constraint: warehouse $i$ can only supply if it is open:
   $$
   \sum_{j \in J} x_{ij} \leq M_i y_i \qquad \forall i \in I
   $$
   where $M_i = \sum_{j \in J} d_j$ (or any sufficiently large number).

3. Variable domains:
   $$
   y_i \in \{0,1\} \qquad \forall i \in I
   $$
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

##### Complete Numerical Formulation

Let $M = 43452$ (sum of all demands).

Minimize:
$$
102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} + 98.71\,y_{S4} + 95.73\,y_{S5} + 99.96\,y_{S6} + 98.16\,y_{S7}
$$
$$
+\,1506.22\,x_{S1,C1} + 70.90\,x_{S1,C2} + 8.44\,x_{S1,C3} + 260.27\,x_{S1,C4} + 197.47\,x_{S1,C5} + 71.71\,x_{S1,C6} + 61.19\,x_{S1,C7}
$$
$$
+\,1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} + 448.68\,x_{S2,C4} + 29.00\,x_{S2,C5} + 1484.91\,x_{S2,C6} + 963.92\,x_{S2,C7}
$$
$$
+\,115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3} + 1324.53\,x_{S3,C4} + 64.99\,x_{S3,C5} + 134.88\,x_{S3,C6} + 2102.83\,x_{S3,C7}
$$
$$
+\,1254.78\,x_{S4,C1} + 1115.63\,x_{S4,C2} + 52.31\,x_{S4,C3} + 1036.16\,x_{S4,C4} + 892.63\,x_{S4,C5} + 1464.04\,x_{S4,C6} + 1383.41\,x_{S4,C7}
$$
$$
+\,42.90\,x_{S5,C1} + 891.01\,x_{S5,C2} + 1013.94\,x_{S5,C3} + 1128.72\,x_{S5,C4} + 58.91\,x_{S5,C5} + 42.89\,x_{S5,C6} + 1570.31\,x_{S5,C7}
$$
$$
+\,0.70\,x_{S6,C1} + 139.46\,x_{S6,C2} + 70.03\,x_{S6,C3} + 79.15\,x_{S6,C4} + 1482.00\,x_{S6,C5} + 0.91\,x_{S6,C6} + 110.46\,x_{S6,C7}
$$
$$
+\,1732.30\,x_{S7,C1} + 1780.44\,x_{S7,C2} + 486.50\,x_{S7,C3} + 523.74\,x_{S7,C4} + 522.08\,x_{S7,C5} + 82.48\,x_{S7,C6} + 826.41\,x_{S7,C7}
$$

Subject to:

For each $j$:
- $x_{S1,j} + x_{S2,j} + x_{S3,j} + x_{S4,j} + x_{S5,j} + x_{S6,j} + x_{S7,j} = d_j$

For each $i$:
- $x_{i,C1} + x_{i,C2} + x_{i,C3} + x_{i,C4} + x_{i,C5} + x_{i,C6} + x_{i,C7} \leq 43452\,y_i$

For all $i, j$:
- $x_{ij} \geq 0$
- $y_i \in \{0,1\}$

All coefficients and identifiers are as retrieved and preserved in source order.