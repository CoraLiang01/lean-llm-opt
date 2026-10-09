##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from warehouse $i \in I$ to musician/band $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is activated, 0 otherwise (binary).

##### Parameters

- $I = \{S1, S2, S3\}$ (Warehouses)
- $J = \{C1, C2, C3\}$ (Musicians/Bands)
- Demand $d_j$ for each $j \in J$:
  - $d_{C1} = 1083$
  - $d_{C2} = 776$
  - $d_{C3} = 16214$
- Fixed cost $f_i$ for each $i \in I$:
  - $f_{S1} = 102.33$
  - $f_{S2} = 94.92$
  - $f_{S3} = 91.83$
- Transportation cost $c_{ij}$ for each $i \in I$, $j \in J$:

|         | C1      | C2      | C3      |
|---------|---------|---------|---------|
| S1      | 1506.22 | 70.9    | 8.44    |
| S2      | 1732.65 | 1780.72 | 567.44  |
| S3      | 115.66  | 100.76  | 64.68   |

- $M = \sum_{j \in J} d_j = 1083 + 776 + 16214 = 18073$ (sufficiently large upper bound for linking constraints)

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
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Full Model with Data

\[
\begin{align*}
\min\quad & 1506.22\,x_{S1,C1} + 70.9\,x_{S1,C2} + 8.44\,x_{S1,C3} \\
&+ 1732.65\,x_{S2,C1} + 1780.72\,x_{S2,C2} + 567.44\,x_{S2,C3} \\
&+ 115.66\,x_{S3,C1} + 100.76\,x_{S3,C2} + 64.68\,x_{S3,C3} \\
&+ 102.33\,y_{S1} + 94.92\,y_{S2} + 91.83\,y_{S3} \\
\text{s.t.}\quad
& x_{S1,C1} + x_{S2,C1} + x_{S3,C1} = 1083 \\
& x_{S1,C2} + x_{S2,C2} + x_{S3,C2} = 776 \\
& x_{S1,C3} + x_{S2,C3} + x_{S3,C3} = 16214 \\
& x_{S1,C1} + x_{S1,C2} + x_{S1,C3} \leq 18073\,y_{S1} \\
& x_{S2,C1} + x_{S2,C2} + x_{S2,C3} \leq 18073\,y_{S2} \\
& x_{S3,C1} + x_{S3,C2} + x_{S3,C3} \leq 18073\,y_{S3} \\
& x_{ij} \geq 0 \quad \forall i \in \{S1, S2, S3\},\, j \in \{C1, C2, C3\} \\
& y_i \in \{0,1\} \quad \forall i \in \{S1, S2, S3\}
\end{align*}
\]

##### Retrieved Information

- Warehouses: S1, S2, S3
- Musicians/Bands: C1, C2, C3
- Demand: {C1: 1083, C2: 776, C3: 16214}
- Fixed Cost: {S1: 102.33, S2: 94.92, S3: 91.83}
- Transportation Cost Matrix:

|         | C1      | C2      | C3      |
|---------|---------|---------|---------|
| S1      | 1506.22 | 70.9    | 8.44    |
| S2      | 1732.65 | 1780.72 | 567.44  |
| S3      | 115.66  | 100.76  | 64.68   |

- $M = 18073$