##### Decision Variables

$x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to branch $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Parameters

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (suppliers)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}\}$ (branches)
- Demands $d_j$ for each branch $j$:
  - $d_{\text{C1}} = 143$
  - $d_{\text{C2}} = 6$
  - $d_{\text{C3}} = 10$
  - $d_{\text{C4}} = 25$
  - $d_{\text{C5}} = 3$
- Fixed costs $f_i$ for each supplier $i$:
  - $f_{\text{S1}} = 97.65000000000001$
  - $f_{\text{S2}} = 99.76000000000001$
  - $f_{\text{S3}} = 100.76$
  - $f_{\text{S4}} = 105.32$
  - $f_{\text{S5}} = 98.88$
- Transportation costs $c_{ij}$ (per unit from supplier $i$ to branch $j$):

|        | C1      | C2      | C3     | C4      | C5     |
|--------|---------|---------|--------|---------|--------|
| S1     | 150.74  | 0.02    | 49.13  | 2080.15 | 426.4  |
| S2     | 233.05  | 97.73   | 49.84  | 1982.39 | 23.96  |
| S3     | 55.68   | 935.61  | 4.03   | 73.09   | 525.32 |
| S4     | 1483.82 | 1801.08 | 112.16 | 816.05  | 107.01 |
| S5     | 1119.47 | 884.31  | 0.08   | 1544.95 | 543.67 |

- $M = \sum_{j \in J} d_j = 143 + 6 + 10 + 25 + 3 = 187$ (big-M for linking $x_{ij}$ and $y_i$)

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
   That is,
   - $\sum_{i \in I} x_{i,\text{C1}} = 143$
   - $\sum_{i \in I} x_{i,\text{C2}} = 6$
   - $\sum_{i \in I} x_{i,\text{C3}} = 10$
   - $\sum_{i \in I} x_{i,\text{C4}} = 25$
   - $\sum_{i \in I} x_{i,\text{C5}} = 3$

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   That is, if $y_i = 0$, then $x_{ij} = 0$ for all $j$.

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

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

Where all parameters ($I$, $J$, $d_j$, $f_i$, $c_{ij}$, $M$) are as listed above, with all data preserved in original order and identifiers.