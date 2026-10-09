Let $x_{ij}$ denote the quantity of beverages shipped from plant $i$ to customer $j$, where $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ and $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$.

Minimize total transportation cost:
\[
\min \;
543.756480860856\, x_{\text{S1},\text{C1}} + 23.685276141764653\, x_{\text{S1},\text{C2}} + 23.676386730773032\, x_{\text{S1},\text{C3}} + 447.75143678673766\, x_{\text{S1},\text{C4}}
\]
\[
+ \; 883.9151090405642\, x_{\text{S2},\text{C1}} + 0.04977684765576961\, x_{\text{S2},\text{C2}} + 0.0350986687216299\, x_{\text{S2},\text{C3}} + 44.45588531711622\, x_{\text{S2},\text{C4}}
\]
\[
+ \; 537.3456896658107\, x_{\text{S3},\text{C1}} + 23.769274659075112\, x_{\text{S3},\text{C2}} + 498.95659249465467\, x_{\text{S3},\text{C3}} + 440.60737890439776\, x_{\text{S3},\text{C4}}
\]
\[
+ \; 1791.493192397229\, x_{\text{S4},\text{C1}} + 68.21633865655126\, x_{\text{S4},\text{C2}} + 1432.4837339656747\, x_{\text{S4},\text{C3}} + 1527.7635425462734\, x_{\text{S4},\text{C4}}
\]

Subject to:

**Demand satisfaction (for each customer):**
\[
x_{\text{S1},j} + x_{\text{S2},j} + x_{\text{S3},j} + x_{\text{S4},j} = d_j \qquad \forall j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}
\]
where
\[
d_{\text{C1}} = 94,\quad d_{\text{C2}} = 39,\quad d_{\text{C3}} = 65,\quad d_{\text{C4}} = 435
\]

That is,
\[
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} = 94
\]
\[
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} = 39
\]
\[
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} = 65
\]
\[
x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} = 435
\]

**Supply capacity (for each plant):**
\[
x_{i,\text{C1}} + x_{i,\text{C2}} + x_{i,\text{C3}} + x_{i,\text{C4}} \leq s_i \qquad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}
\]
where
\[
s_{\text{S1}} = 2531,\quad s_{\text{S2}} = 20,\quad s_{\text{S3}} = 210,\quad s_{\text{S4}} = 241
\]

That is,
\[
x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} \leq 2531
\]
\[
x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} \leq 20
\]
\[
x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} \leq 210
\]
\[
x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} \leq 241
\]

**Nonnegativity:**
\[
x_{ij} \geq 0 \qquad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\},\; j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}
\]

**Variables:**
\[
x_{ij} \geq 0 \text{ (continuous)},\quad \forall i,j
\]

**Summary of sets and parameters:**

- Plants: S1, S2, S3, S4
- Customers: C1, C2, C3, C4
- Demands: C1: 94, C2: 39, C3: 65, C4: 435
- Supply capacities: S1: 2531, S2: 20, S3: 210, S4: 241
- Transportation costs per unit: as given in transportation_costs.csv above

This is a complete numerical linear programming model for BrewCo's transportation problem.