##### Decision Variables

$x_{ij} \geq 0$: Quantity supplied from supplier $i \in I$ to supermarket $j \in J$ (continuous).  
$y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{\text{S1}, \text{S2}\}$ (Suppliers)
- $J = \{\text{C1}, \text{C2}\}$ (Supermarkets)

- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$

- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$

- Transportation costs (per unit):

|           | C1      | C2     |
|-----------|---------|--------|
| S1        | 2358.39 | 1492.08|
| S2        | 0.0700  | 52.32  |

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

where $c_{ij}$ is the per-unit transportation cost from supplier $i$ to supermarket $j$.

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   where $M = \sum_{j \in J} d_j = 144 + 216 = 360$ (a valid upper bound since there are no explicit supplier capacity limits).

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
\min\ & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} \\
& + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216 \\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}} \\
& x_{ij} \geq 0 \quad \forall i \in \{\text{S1},\text{S2}\},\, j \in \{\text{C1},\text{C2}\} \\
& y_i \in \{0,1\} \quad \forall i \in \{\text{S1},\text{S2}\}
\end{align*}
\]

##### Retrieved Information

- Suppliers: S1, S2
- Supermarkets: C1, C2
- Demand: C1 = 144, C2 = 216
- Fixed costs: S1 = 105.97, S2 = 85.31
- Transportation costs:
    - S1: C1 = 2358.39, C2 = 1492.08
    - S2: C1 = 0.0700, C2 = 52.32
- $M = 360$