##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: whether supplier $i$ is activated.

##### Sets

- $I = \{\text{S1}, \text{S2}\}$ (suppliers)
- $J = \{\text{C1}, \text{C2}\}$ (supermarkets)

##### Parameters

- Demand:
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
- $M = \sum_{j \in J} d_j = 144 + 216 = 360$ (sufficiently large upper bound for each supplier's total shipment)

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each supermarket $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]
   - $\sum_{i \in \{\text{S1},\text{S2}\}} x_{i,\text{C1}} = 144$
   - $\sum_{i \in \{\text{S1},\text{S2}\}} x_{i,\text{C2}} = 216$

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   - $\sum_{j \in \{\text{C1},\text{C2}\}} x_{\text{S1},j} \leq 360 y_{\text{S1}}$
   - $\sum_{j \in \{\text{C1},\text{C2}\}} x_{\text{S2},j} \leq 360 y_{\text{S2}}$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Parameter Tables

- **Demand:**

| Customer | Demand |
|----------|--------|
| C1       | 144    |
| C2       | 216    |

- **Fixed Costs:**

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 105.97     |
| S2       | 85.31      |

- **Transportation Costs:**

| Supplier | C1      | C2     |
|----------|---------|--------|
| S1       | 2358.39 | 1492.08|
| S2       | 0.07    | 52.32  |

##### Complete Mathematical Model

\[
\begin{align*}
\min\quad & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} \\
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