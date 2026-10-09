##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i$ to supermarket $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated, 0 otherwise (binary).

##### Parameters

- Suppliers $I = \{\text{S1}, \text{S2}\}$
- Supermarkets $J = \{\text{C1}, \text{C2}\}$

- Demands:
  - $d_{\text{C1}} = 144$
  - $d_{\text{C2}} = 216$

- Fixed opening costs:
  - $f_{\text{S1}} = 105.97$
  - $f_{\text{S2}} = 85.31$

- Transportation costs per unit:
  - $c_{\text{S1},\text{C1}} = 2358.39$
  - $c_{\text{S1},\text{C2}} = 1492.08$
  - $c_{\text{S2},\text{C1}} = 0.07$
  - $c_{\text{S2},\text{C2}} = 52.32$

- $M = d_{\text{C1}} + d_{\text{C2}} = 360$ (sufficiently large upper bound for linking constraints)

##### Objective Function

\[
\min \left( 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \right)
\]

##### Constraints

1. **Demand satisfaction (each supermarket's demand must be met):**
   - $x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144$
   - $x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216$

2. **Supplier activation (inactive suppliers cannot ship goods):**
   - $x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}}$
   - $x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}}$

3. **Variable domains:**
   - $x_{ij} \geq 0$ for all $i \in I$, $j \in J$
   - $y_i \in \{0,1\}$ for all $i \in I$

##### Parameter Tables

**Demands:**

| Customer | Demand |
|----------|--------|
| C1       | 144    |
| C2       | 216    |

**Fixed Opening Costs:**

| Supplier | Fixed Cost |
|----------|------------|
| S1       | 105.97     |
| S2       | 85.31      |

**Transportation Costs:**

| Supplier | C1      | C2     |
|----------|---------|--------|
| S1       | 2358.39 | 1492.08|
| S2       | 0.07    | 52.32  |

##### Sets

- $I = \{\text{S1}, \text{S2}\}$
- $J = \{\text{C1}, \text{C2}\}$

##### Complete Model

\[
\begin{align*}
\min\ & 2358.39\,x_{\text{S1},\text{C1}} + 1492.08\,x_{\text{S1},\text{C2}} + 0.07\,x_{\text{S2},\text{C1}} + 52.32\,x_{\text{S2},\text{C2}} \\
&\quad + 105.97\,y_{\text{S1}} + 85.31\,y_{\text{S2}} \\
\text{s.t.}\quad
& x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} = 144 \\
& x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} = 216 \\
& x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} \leq 360\,y_{\text{S1}} \\
& x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} \leq 360\,y_{\text{S2}} \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]