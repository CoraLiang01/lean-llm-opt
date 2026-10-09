##### Decision Variables

- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to supermarket $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise.

##### Parameters

- $I = \{\text{S1}, \text{S2}\}$ (suppliers)
- $J = \{\text{C1}, \text{C2}\}$ (supermarkets)
- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$
- Fixed costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$
- Transportation costs per unit:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$
- $M = \sum_{j \in J} d_j = 144 + 216 = 360$

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
   - For C1: $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - For C2: $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]
   - For S1: $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360 y_{\text{S1}}$
   - For S2: $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360 y_{\text{S2}}$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Parameter Tables

**Demands:**

| Customer | Demand |
|----------|--------|
| C1       | 144    |
| C2       | 216    |

**Fixed Costs:**

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 105.97     |
| S2       | 85.31      |

**Transportation Costs:**

| Supplier | C1      | C2     |
|----------|---------|--------|
| S1       | 2358.39 | 1492.08|
| S2       | 0.07    | 52.32  |

**Big M:** $M = 360$

##### Sets

- $I = \{\text{S1}, \text{S2}\}$
- $J = \{\text{C1}, \text{C2}\}$

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
& x_{ij} \geq 0 \quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\} \quad \forall i \in I
\end{align*}
\]