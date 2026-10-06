##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Parameters

- $I = \{S1, S2, S3, S4, S5\}$ (suppliers, in source order)
- $J = \{C1, C2, C3, C4, C5\}$ (branches, in source order)

- Demand vector $d_j$ (units required at each branch):
  - $d_{C1} = 143$
  - $d_{C2} = 6$
  - $d_{C3} = 10$
  - $d_{C4} = 25$
  - $d_{C5} = 3$

- Fixed opening cost $f_i$ for each supplier:
  - $f_{S1} = 97.65$
  - $f_{S2} = 99.76$
  - $f_{S3} = 100.76$
  - $f_{S4} = 105.32$
  - $f_{S5} = 98.88$

- Transportation cost matrix $c_{ij}$ (cost per unit from supplier $i$ to branch $j$):

|         | C1      | C2      | C3     | C4      | C5     |
|---------|---------|---------|--------|---------|--------|
| S1      | 150.74  | 0.02    | 49.13  | 2080.15 | 426.4  |
| S2      | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96  |
| S3      | 55.68   | 935.61  | 4.03   | 73.09   | 525.32 |
| S4      | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01 |
| S5      | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67 |

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

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (sufficiently large upper bound).

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 187\, y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

Where all parameters ($c_{ij}$, $f_i$, $d_j$) are as listed above, and $I$, $J$ are the supplier and branch sets in source order.