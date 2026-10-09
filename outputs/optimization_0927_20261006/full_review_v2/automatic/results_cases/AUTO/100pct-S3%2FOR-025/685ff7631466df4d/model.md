##### Decision Variables

$x_{ij} \geq 0$: fraction of supermarket $j \in J$'s demand supplied by supplier $i \in I$ (continuous, $0 \leq x_{ij} \leq 1$).
$y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- Suppliers $I = \{\text{S1}, \text{S2}\}$
- Supermarkets $J = \{\text{C1}, \text{C2}\}$
- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$
- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$
- Transportation costs (per unit):
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} d_j + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. Demand satisfaction for each supermarket:
   \[
   \sum_{i \in I} x_{ij} = 1, \quad \forall j \in J
   \]
   (Each supermarket's demand must be fully met.)

2. Supplier activation logic:
   \[
   x_{ij} \leq y_i, \quad \forall i \in I,\, j \in J
   \]
   (A supplier can only supply if it is activated.)

3. Variable domains:
   \[
   x_{ij} \geq 0, \quad x_{ij} \leq 1, \quad y_i \in \{0,1\}
   \]

##### Explicit Data

- $I = \{\text{S1}, \text{S2}\}$
- $J = \{\text{C1}, \text{C2}\}$
- $d_{\text{C1}} = 144$, $d_{\text{C2}} = 216$
- $f_{\text{S1}} = 105.97$, $f_{\text{S2}} = 85.31$
- $c_{ij}$ matrix:

|        | C1      | C2      |
|--------|---------|---------|
| S1     | 2358.39 | 1492.08 |
| S2     | 0.07    | 52.32   |

##### Full Model

\[
\begin{align*}
\min\ & 2358.39\, x_{\text{S1},\text{C1}} \cdot 144 + 1492.08\, x_{\text{S1},\text{C2}} \cdot 216 \\
     & +\ 0.07\, x_{\text{S2},\text{C1}} \cdot 144 + 52.32\, x_{\text{S2},\text{C2}} \cdot 216 \\
     & +\ 105.97\, y_{\text{S1}} + 85.31\, y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 1 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 1 \\
& x_{\text{S1},\text{C1}} \leq y_{\text{S1}} \\
& x_{\text{S1},\text{C2}} \leq y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} \leq y_{\text{S2}} \\
& x_{\text{S2},\text{C2}} \leq y_{\text{S2}} \\
& 0 \leq x_{ij} \leq 1,\quad y_i \in \{0,1\}
\end{align*}
\]