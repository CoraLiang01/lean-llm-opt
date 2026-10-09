#### Abstract Mathematical Model

Let:
- $\mathcal{I}$: index set of all products classified under ‘Fashion’ (from column ‘Product Name’).
- For each $i \in \mathcal{I}$:
    - $A_i$: revenue per unit of product $i$ (from column ‘Revenue’).
    - $d_i$: deterministic demand for product $i$ (from column ‘Demand’).
    - $I_i$: initial inventory for product $i$ (from column ‘Initial Inventory’).
    - $x_i$: integer decision variable, number of units of product $i$ to fulfill.

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (SupermarketSales.csv)
    - Index set $\mathcal{I}$: All rows where ‘Product Name’ has prefix "Fashion"
    - $A_i$: column ‘Revenue’
    - $d_i$: column ‘Demand’
    - $I_i$: column ‘Initial Inventory’