#### Symbolic Optimization Model

Let:
- $\mathcal{F}$ = set of all products classified as ‘Fashion’ (indexed by $i$)
- $A_i$ = revenue per unit of product $i$ (parameter from column ‘Revenue’)
- $d_i$ = deterministic demand for product $i$ (parameter from column ‘Demand’)
- $I_i$ = initial inventory for product $i$ (parameter from column ‘Initial Inventory’)
- $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in \mathcal{F}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in \mathcal{F} \\
& x_i \leq I_i, \quad \forall i \in \mathcal{F} \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \mathcal{F}
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from SupermarketSales.csv)
    - Index set $\mathcal{F}$: All rows where ‘Product Name’ has prefix "Fashion"
    - Parameter $A_i$: column ‘Revenue’
    - Parameter $d_i$: column ‘Demand’
    - Parameter $I_i$: column ‘Initial Inventory’