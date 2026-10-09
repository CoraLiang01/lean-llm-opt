##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to branch $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated (binary).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each branch $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
2. **Supplier activation logic:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j = 187$ is a valid upper bound on total shipments from any supplier.
3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \qquad y_i \in \{0,1\}
   \]

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (suppliers)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$ (branches)

##### Parameters

- **Branch demands $d_j$:**

| Branch | Demand ($d_j$) |
|--------|---------------|
| C1     | 143           |
| C2     | 6             |
| C3     | 10            |
| C4     | 25            |
| C5     | 3             |

- **Supplier fixed opening costs $f_i$:**

| Supplier | Fixed Cost ($f_i$) |
|----------|--------------------|
| S1       | 97.65              |
| S2       | 99.76              |
| S3       | 100.76             |
| S4       | 105.32             |
| S5       | 98.88              |

- **Transportation costs $c_{ij}$ (per unit from supplier $i$ to branch $j$):**

| Supplier | C1      | C2      | C3    | C4      | C5     |
|----------|---------|---------|-------|---------|--------|
| S1       | 150.74  | 0.02    | 49.13 | 2080.15 | 426.4  |
| S2       | 233.05  | 97.73   | 49.84 | 1982.39 | 23.96  |
| S3       | 55.68   | 935.61  | 4.03  | 73.09   | 525.32 |
| S4       | 1483.82 | 1801.08 | 112.16| 816.05  | 107.01 |
| S5       | 1119.47 | 884.31  | 0.08  | 1544.95 | 543.67 |

- **Big-M parameter:**  
  $M = 143 + 6 + 10 + 25 + 3 = 187$

##### Full Model

\[
\begin{align*}
\min \quad & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.} \quad & \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \qquad \forall i \in I \\
& x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \qquad \forall i \in I
\end{align*}
\]

where all parameters and sets are as specified above.