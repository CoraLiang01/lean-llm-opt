##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i
\]

##### Constraints

1. Branch demand:  
   \[
   \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J
   \]
2. Supplier activation:  
   \[
   \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j\in J} d_j = 187$
3. Domains:  
   \[
   x_{ij} \geq 0 \text{ (continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters

- Suppliers: $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Branches: $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$

###### Demand vector $d_j$ (from demand.csv):

| Branch | Demand $d_j$ |
|--------|--------------|
| C1     | 143          |
| C2     | 6            |
| C3     | 10           |
| C4     | 25           |
| C5     | 3            |

###### Fixed opening costs $f_i$ (from fixed_cost.csv):

| Supplier | Fixed Cost $f_i$ |
|----------|------------------|
| S1       | 97.65            |
| S2       | 99.76            |
| S3       | 100.76           |
| S4       | 105.32           |
| S5       | 98.88            |

###### Transportation cost matrix $c_{ij}$ (from transportation_costs.csv):

| Supplier | C1      | C2      | C3    | C4      | C5     |
|----------|---------|---------|-------|---------|--------|
| S1       | 150.74  | 0.02    | 49.13 | 2080.15 | 426.4  |
| S2       | 233.05  | 97.73   | 49.84 | 1982.39 | 23.96  |
| S3       | 55.68   | 935.61  | 4.03  | 73.09   | 525.32 |
| S4       | 1483.82 | 1801.08 | 112.16| 816.05  | 107.01 |
| S5       | 1119.47 | 884.31  | 0.08  | 1544.95 | 543.67 |

##### Summary of Sets and Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$
- $d = [143, 6, 10, 25, 3]$
- $f = [97.65, 99.76, 100.76, 105.32, 98.88]$
- $c =$
  \[
  \begin{bmatrix}
  150.74 & 0.02 & 49.13 & 2080.15 & 426.4 \\
  233.05 & 97.73 & 49.84 & 1982.39 & 23.96 \\
  55.68 & 935.61 & 4.03 & 73.09 & 525.32 \\
  1483.82 & 1801.08 & 112.16 & 816.05 & 107.01 \\
  1119.47 & 884.31 & 0.08 & 1544.95 & 543.67 \\
  \end{bmatrix}
  \]
- $M = 187$

##### Complete Mathematical Model

\[
\begin{align*}
\min \quad & \sum_{i\in I}\sum_{j\in J} c_{ij}x_{ij} + \sum_{i\in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i\in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j\in J} x_{ij} \leq M y_i, \quad \forall i \in I \\
& x_{ij} \geq 0, \quad \forall i \in I, j \in J \\
& y_i \in \{0,1\}, \quad \forall i \in I
\end{align*}
\]

where all parameters are as listed above.