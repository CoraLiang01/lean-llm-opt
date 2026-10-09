#### Symbolic Mathematical Model

Let:
- $\mathcal{I}$ = index set of all products classified under ‘Fashion’ (from the data).
- For each $i \in \mathcal{I}$:
    - $A_i$ = revenue per unit of product $i$ (parameter).
    - $d_i$ = deterministic demand for product $i$ (parameter).
    - $I_i$ = initial inventory for product $i$ (parameter).
    - $x_i$ = number of units of product $i$ to fulfill (decision variable).

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \geq 0, \quad x_i \in \mathbb{Z}, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (SupermarketSales.csv)
    - Index set $\mathcal{I}$: All rows where [Product Name] starts with "Fashion"
    - $A_i$: [Revenue]
    - $d_i$: [Demand]
    - $I_i$: [Initial Inventory]
    - $x_i$: Decision variable for each $i \in \mathcal{I}$

All parameters are mapped directly from the specified columns in the returned table.