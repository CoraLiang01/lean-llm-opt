##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- Warehouses $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}\}$
- Musicians/Bands $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}\}$

- Demands:
  - $d_{\text{C1}} = 1083$
  - $d_{\text{C2}} = 776$
  - $d_{\text{C3}} = 16214$
  - $d_{\text{C4}} = 553$
  - $d_{\text{C5}} = 17106$
  - $d_{\text{C6}} = 594$
  - $d_{\text{C7}} = 732$

- Fixed costs:
  - $f_{\text{S1}} = 102.33$
  - $f_{\text{S2}} = 94.92$
  - $f_{\text{S3}} = 91.83$
  - $f_{\text{S4}} = 98.71$
  - $f_{\text{S5}} = 95.73$
  - $f_{\text{S6}} = 99.96$
  - $f_{\text{S7}} = 98.16$

- Transportation costs $c_{ij}$ (per unit):

|         | C1      | C2      | C3      | C4      | C5      | C6      | C7      |
|---------|---------|---------|---------|---------|---------|---------|---------|
| S1      | 1506.22 | 70.90   | 8.44    | 260.27  | 197.47  | 71.71   | 61.19   |
| S2      | 1732.65 | 1780.72 | 567.44  | 448.68  | 29.00   | 1484.91 | 963.92  |
| S3      | 115.66  | 100.76  | 64.68   | 1324.53 | 64.99   | 134.88  | 2102.83 |
| S4      | 1254.78 | 1115.63 | 52.31   | 1036.16 | 892.63  | 1464.04 | 1383.41 |
| S5      | 42.90   | 891.01  | 1013.94 | 1128.72 | 58.91   | 42.89   | 1570.31 |
| S6      | 0.70    | 139.46  | 70.03   | 79.15   | 1482.00 | 0.91    | 110.46  |
| S7      | 1732.30 | 1780.44 | 486.50  | 523.74  | 522.08  | 82.48   | 826.41  |

- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 + 553 + 17106 + 594 + 732 = 36458$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each musician/band $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Warehouse activation:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = 36458$.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### All Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}\}$
- $d_j$ as above
- $f_i$ as above
- $c_{ij}$ as above
- $M = 36458$