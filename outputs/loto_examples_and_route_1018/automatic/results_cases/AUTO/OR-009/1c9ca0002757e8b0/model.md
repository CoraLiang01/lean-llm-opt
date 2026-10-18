Let $x_{sc}$ denote the quantity of beverages shipped from supply plant $s$ to customer $c$.

**Sets and Parameters (from data):**

- Supply plants $s \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- Customers $c \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

- Customer demands:
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$

- Supply capacities:
  - $u_{\text{S1}} = 2531$
  - $u_{\text{S2}} = 20$
  - $u_{\text{S3}} = 210$
  - $u_{\text{S4}} = 241$

- Transportation costs $c_{sc}$:

|        | C1              | C2                | C3                | C4                |
|--------|-----------------|-------------------|-------------------|-------------------|
| S1     | 543.756480860856  | 23.685276141764653 | 23.676386730773032 | 447.75143678673766 |
| S2     | 883.9151090405642 | 0.04977684765576961 | 0.0350986687216299 | 44.45588531711622  |
| S3     | 537.3456896658107 | 23.769274659075112  | 498.95659249465467 | 440.60737890439776 |
| S4     | 1791.493192397229 | 68.21633865655126   | 1432.4837339656747 | 1527.7635425462734 |

---

**Mathematical Model:**

Minimize total transportation cost:
$$
\min \sum_{s \in \{\text{S1},\text{S2},\text{S3},\text{S4}\}} \sum_{c \in \{\text{C1},\text{C2},\text{C3},\text{C4}\}} c_{sc} \cdot x_{sc}
$$

Subject to:

**1. Demand satisfaction (for each customer):**
\[
\begin{align*}
x_{\text{S1},c} + x_{\text{S2},c} + x_{\text{S3},c} + x_{\text{S4},c} = d_c, \quad \forall c \in \{\text{C1},\text{C2},\text{C3},\text{C4}\}
\end{align*}
\]
That is,
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} &= 94 \\
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} &= 39 \\
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} &= 65 \\
x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} &= 435 \\
\end{align*}
\]

**2. Supply capacity (for each plant):**
\[
\begin{align*}
x_{s,\text{C1}} + x_{s,\text{C2}} + x_{s,\text{C3}} + x_{s,\text{C4}} \leq u_s, \quad \forall s \in \{\text{S1},\text{S2},\text{S3},\text{S4}\}
\end{align*}
\]
That is,
\[
\begin{align*}
x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} &\leq 2531 \\
x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} &\leq 20 \\
x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} &\leq 210 \\
x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} &\leq 241 \\
\end{align*}
\]

**3. Nonnegativity:**
\[
x_{sc} \geq 0 \quad \forall s, c
\]

**4. Variable domain:**
\[
x_{sc} \in \mathbb{R}_{\geq 0} \quad \forall s, c
\]

---

**All data used:**

- Customers: C1, C2, C3, C4
- Demands: 94, 39, 65, 435
- Plants: S1, S2, S3, S4
- Capacities: 2531, 20, 210, 241
- Transportation costs: as in the table above

**Decision variables:** $x_{sc}$ = quantity shipped from plant $s$ to customer $c$, continuous and nonnegative.