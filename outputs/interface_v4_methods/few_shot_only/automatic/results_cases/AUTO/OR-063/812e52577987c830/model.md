##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether warehouse $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij} x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction:  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Warehouse activation:  
   \[
   \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j\in J} d_j = 37,\!058$.
3. Domains:  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Sets and Parameters

- Warehouses $I = \{S1, S2, S3, S4, S5, S6, S7\}$
- Musicians/Bands $J = \{C1, C2, C3, C4, C5, C6, C7\}$

- Demands $d_j$:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16,\!214$
  - $d_{C4} = 553$
  - $d_{C5} = 17,\!106$
  - $d_{C6} = 594$
  - $d_{C7} = 732$

- Warehouse fixed costs $f_i$:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$
  - $f_{S4} = 98.71$
  - $f_{S5} = 95.73$
  - $f_{S6} = 99.96$
  - $f_{S7} = 98.16$

- Transportation costs $c_{ij}$:

|        | C1      | C2      | C3      | C4      | C5      | C6      | C7      |
|--------|---------|---------|---------|---------|---------|---------|---------|
| S1     | 1506.22 | 70.90   | 8.44    | 260.27  | 197.47  | 71.71   | 61.19   |
| S2     | 1732.65 | 1780.72 | 567.44  | 448.68  | 29.00   | 1484.91 | 963.92  |
| S3     | 115.66  | 100.76  | 64.68   | 1324.53 | 64.99   | 134.88  | 2102.83 |
| S4     | 1254.78 | 1115.63 | 52.31   | 1036.16 | 892.63  | 1464.04 | 1383.41 |
| S5     | 42.90   | 891.01  | 1013.94 | 1128.72 | 58.91   | 42.89   | 1570.31 |
| S6     | 0.70    | 139.46  | 70.03   | 79.15   | 1482.00 | 0.91    | 110.46  |
| S7     | 1732.30 | 1780.44 | 486.50  | 523.74  | 522.08  | 82.48   | 826.41  |

- $M = 37,\!058$ (sum of all demands)

##### Full Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

where all parameters are as listed above.