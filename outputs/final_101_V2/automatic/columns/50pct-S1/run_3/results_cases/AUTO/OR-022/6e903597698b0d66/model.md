##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from supplier (facility) $i \in I$ to branch (customer) $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if supplier (facility) $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (set of suppliers/facilities)
- $J = \{C1, C2, C3, C4, C5\}$ (set of branches/customers)

- Demand $d_j$ for each branch $j$:
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Fixed opening cost $f_i$ for each supplier $i$:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation cost $c_{ij}$ from supplier $i$ to branch $j$:

|         | C1      | C2      | C3    | C4      | C5     |
|---------|---------|---------|-------|---------|--------|
| S1      | 150.74  | 0.02    | 49.13 | 2080.15 | 426.4  |
| S2      | 233.05  | 97.73   | 49.84 | 1982.39 | 23.96  |
| S3      | 55.68   | 935.61  | 4.03  | 73.09   | 525.32 |
| S4      | 1483.82 | 1801.08 | 112.16| 816.05  | 107.01 |
| S5      | 1119.47 | 884.31  | 0.08  | 1544.95 | 543.67 |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (sufficiently large upper bound for linking constraints)

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

##### Constraints

1. **Demand satisfaction:**  
   For each branch $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation (linking):**  
   For each supplier $i \in I$,
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

##### All Parameters (Vectors and Matrices)

- $I = \{S1, S2, S3, S4, S5\}$
- $J = \{C1, C2, C3, C4, C5\}$
- $d = [143, 6, 10, 25, 3]$ (ordered as $[C1, C2, C3, C4, C5]$)
- $f = [97.65, 99.76, 100.76, 105.32, 98.88]$ (ordered as $[S1, S2, S3, S4, S5]$)
- $C =$
\[
\begin{bmatrix}
150.74 & 0.02 & 49.13 & 2080.15 & 426.4 \\
233.05 & 97.73 & 49.84 & 1982.39 & 23.96 \\
55.68 & 935.61 & 4.03 & 73.09 & 525.32 \\
1483.82 & 1801.08 & 112.16 & 816.05 & 107.01 \\
1119.47 & 884.31 & 0.08 & 1544.95 & 543.67 \\
\end{bmatrix}
\]
(rows: $S1$–$S5$, columns: $C1$–$C5$)

- $M = 187$

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
\end{align*}
\]

Where all parameters and sets are as listed above, with all vectors and matrices explicitly provided.