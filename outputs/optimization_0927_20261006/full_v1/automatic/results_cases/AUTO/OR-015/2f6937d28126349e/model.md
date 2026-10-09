#### Abstract Mathematical Optimization Model

Let:

- $\mathcal{I}$: Index set of all products classified under ‘Aalop’ (from column ‘Product Name’ in table_id file_0_view_0).
- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id file_0_view_0).
    - $d_i$: Demand for product $i$ over the sales horizon (from column ‘Demand’ in table_id file_0_view_0).
    - $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’ in table_id file_0_view_0).
    - $x_i$: Decision variable; number of units of product $i$ to fulfill.

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq I_i, \quad \forall i \in \mathcal{I} \\
& x_i \leq d_i, \quad \forall i \in \mathcal{I} \\
& x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
\end{align*}
\]

#### Data Mapping

- Table: RestaurantSalesreport.csv (table_id: file_0_view_0)
    - Index set $\mathcal{I}$: All rows where ‘Product Name’ has prefix ‘Aalop’
    - $A_i$: ‘Revenue’ column
    - $d_i$: ‘Demand’ column
    - $I_i$: ‘Initial Inventory’ column