##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from facility (supplier) $i \in I$ to customer (branch) $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether facility (supplier) $i$ is activated (open).

##### Parameters

- Facilities (Suppliers): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Customers (Branches): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$

- Demand $d_j$ for each customer $j$:
  - $d_{\text{C1}} = 143$
  - $d_{\text{C2}} = 6$
  - $d_{\text{C3}} = 10$
  - $d_{\text{C4}} = 25$
  - $d_{\text{C5}} = 3$

- Fixed opening cost $f_i$ for each facility $i$:
  - $f_{\text{S1}} = 97.65$
  - $f_{\text{S2}} = 99.76$
  - $f_{\text{S3}} = 100.76$
  - $f_{\text{S4}} = 105.32$
  - $f_{\text{S5}} = 98.88$

- Transportation cost $c_{ij}$ from facility $i$ to customer $j$:

|        | C1      | C2      | C3    | C4      | C5     |
|--------|---------|---------|-------|---------|--------|
| S1     | 150.74  | 0.02    | 49.13 | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73   | 49.84 | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61  | 4.03  | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08 | 112.16| 816.05  | 107.01 |
| S5     | 1119.47 | 884.31  | 0.08  | 1544.95 | 543.67 |

Let $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (a valid upper bound for total shipments from any facility, since there are no explicit capacity limits).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Facility activation:**  
   For each facility $i \in I$,
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

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$
- $d = [143,\, 6,\, 10,\, 25,\, 3]$
- $f = [97.65,\, 99.76,\, 100.76,\, 105.32,\, 98.88]$
- $C = \begin{bmatrix}
150.74 & 0.02 & 49.13 & 2080.15 & 426.4 \\
233.05 & 97.73 & 49.84 & 1982.39 & 23.96 \\
55.68 & 935.61 & 4.03 & 73.09 & 525.32 \\
1483.82 & 1801.08 & 112.16 & 816.05 & 107.01 \\
1119.47 & 884.31 & 0.08 & 1544.95 & 543.67
\end{bmatrix}$

- $M = 187$

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]

where all parameters, indices, and values are as listed above, and all data is preserved from the original CSV sources.