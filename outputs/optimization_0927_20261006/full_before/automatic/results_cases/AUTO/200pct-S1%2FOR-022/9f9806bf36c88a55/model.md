##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from facility (supplier) $i \in I$ to customer (branch) $j \in J$ (continuous).  
$y_i \in \{0,1\}$: whether facility (supplier) $i$ is activated (open).

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where:
- $c_{ij}$: transportation cost per unit from facility $i$ to customer $j$
- $f_i$: fixed opening cost for facility $i$

##### Constraints

1. **Demand Satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   where $d_j$ is the demand of customer $j$.

2. **Facility Activation:**  
   For each facility $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j = 187$ (the total demand), ensuring that no goods are shipped from a closed facility.

3. **Variable Domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Sets and Parameters

- Facilities (Suppliers): $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Customers (Branches): $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$

###### Demand Vector

\[
\begin{align*}
d_{\text{C1}} &= 143 \\
d_{\text{C2}} &= 6 \\
d_{\text{C3}} &= 10 \\
d_{\text{C4}} &= 25 \\
d_{\text{C5}} &= 3 \\
\end{align*}
\]

###### Fixed Opening Cost Vector

\[
\begin{align*}
f_{\text{S1}} &= 97.65 \\
f_{\text{S2}} &= 99.76 \\
f_{\text{S3}} &= 100.76 \\
f_{\text{S4}} &= 105.32 \\
f_{\text{S5}} &= 98.88 \\
\end{align*}
\]

###### Transportation Cost Matrix $[c_{ij}]$

|        | C1      | C2     | C3    | C4      | C5     |
|--------|---------|--------|-------|---------|--------|
| S1     | 150.74  | 0.02   | 49.13 | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73  | 49.84 | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61 | 4.03  | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08| 112.16| 816.05  | 107.01 |
| S5     | 1119.47 | 884.31 | 0.08  | 1544.95 | 543.67 |

###### Big-M Value

\[
M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187
\]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq M y_i \quad \forall i \in I \\
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I \\
\end{align*}
\]

with all parameters as listed above, and all data preserved in original source order and identifiers.