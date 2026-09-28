##### Sets and Indices
- Suppliers: $i \in \{\text{S1}, \text{S2}\}$
- Supermarkets: $j \in \{\text{C1}, \text{C2}\}$

##### Parameters (from CSVs)
- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$
- Transportation costs per unit:
  - $t_{\text{S1},\text{C1}} = 2358.39$
  - $t_{\text{S1},\text{C2}} = 1492.08$
  - $t_{\text{S2},\text{C1}} = 0.07$
  - $t_{\text{S2},\text{C2}} = 52.32$
- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$

##### Decision Variables
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise
- $x_{ij} \geq 0$: amount supplied from supplier $i$ to supermarket $j$

##### Objective Function
Minimize total cost:
$$
\min \sum_{i \in \{\text{S1}, \text{S2}\}} f_i y_i + \sum_{i \in \{\text{S1}, \text{S2}\}} \sum_{j \in \{\text{C1}, \text{C2}\}} t_{ij} x_{ij}
$$
That is,
$$
\min \left[ 105.97\, y_{\text{S1}} + 85.31\, y_{\text{S2}} + 2358.39\, x_{\text{S1},\text{C1}} + 1492.08\, x_{\text{S1},\text{C2}} + 0.07\, x_{\text{S2},\text{C1}} + 52.32\, x_{\text{S2},\text{C2}} \right]
$$

##### Constraints

1. **Demand satisfaction for each supermarket:**
   $$
   x_{\text{S1},j} + x_{\text{S2},j} = d_j, \quad \forall j \in \{\text{C1}, \text{C2}\}
   $$
   Explicitly:
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Supplier activation:**
   $$
   x_{ij} \leq d_j\, y_i, \quad \forall i \in \{\text{S1}, \text{S2}\},\; j \in \{\text{C1}, \text{C2}\}
   $$
   (If supplier $i$ is not activated, it cannot supply any supermarket.)

3. **Variable domains:**
   $$
   y_i \in \{0,1\}, \quad \forall i \in \{\text{S1}, \text{S2}\}
   $$
   $$
   x_{ij} \geq 0, \quad \forall i \in \{\text{S1}, \text{S2}\},\; j \in \{\text{C1}, \text{C2}\}
   $$

##### Complete Model

Minimize:
$$
105.97\, y_{\text{S1}} + 85.31\, y_{\text{S2}} + 2358.39\, x_{\text{S1},\text{C1}} + 1492.08\, x_{\text{S1},\text{C2}} + 0.07\, x_{\text{S2},\text{C1}} + 52.32\, x_{\text{S2},\text{C2}}
$$

Subject to:
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} &= 144 \\
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} &= 216 \\
x_{\text{S1},\text{C1}} &\leq 144\, y_{\text{S1}} \\
x_{\text{S1},\text{C2}} &\leq 216\, y_{\text{S1}} \\
x_{\text{S2},\text{C1}} &\leq 144\, y_{\text{S2}} \\
x_{\text{S2},\text{C2}} &\leq 216\, y_{\text{S2}} \\
y_{\text{S1}},\, y_{\text{S2}} &\in \{0,1\} \\
x_{ij} &\geq 0 \quad \forall i,j \\
\end{align*}
\]

All coefficients, identifiers, and constraints are as retrieved and required.