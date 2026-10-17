Let $x_{ij}$ denote the quantity of beverages shipped from plant $i$ to customer $j$, where $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ and $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$.

**Parameters:**

- Supply capacities:
  - S1: $2531$
  - S2: $20$
  - S3: $210$
  - S4: $241$

- Customer demands:
  - C1: $94$
  - C2: $39$
  - C3: $65$
  - C4: $435$

- Transportation costs per unit ($c_{ij}$):

  |        | C1                | C2                | C3                | C4                |
  |--------|-------------------|-------------------|-------------------|-------------------|
  | S1     | 543.756480860856  | 23.685276141764653| 23.676386730773032| 447.75143678673766|
  | S2     | 883.9151090405642 | 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
  | S3     | 537.3456896658107 | 23.769274659075112| 498.95659249465467| 440.60737890439776|
  | S4     | 1791.493192397229 | 68.21633865655126 | 1432.4837339656747| 1527.7635425462734|

---

**Mathematical Model:**

Minimize total transportation cost:
$$
\min \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}} \sum_{j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}} c_{ij} \, x_{ij}
$$

Subject to:

**1. Supply capacity at each plant:**
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} &\leq 2531 \\
x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} &\leq 20 \\
x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} &\leq 210 \\
x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} &\leq 241 \\
\end{align*}
\]

**2. Demand satisfaction at each customer:**
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} &= 94 \\
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} &= 39 \\
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} &= 65 \\
x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} &= 435 \\
\end{align*}
\]

**3. Nonnegativity:**
\[
x_{ij} \geq 0 \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\},\; j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}
\]

---

**Where:**

- $x_{ij}$: quantity shipped from plant $i$ to customer $j$ (continuous, $\geq 0$)
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (see table above)

All identifiers, coefficients, and constraints are as retrieved and in source order.