Mathematical Optimization Model

Index Sets:
Let $\mathcal{I}$ be the set of all products with Product_Name classified under ‘Books’ in table_id file_0_view_0.

Parameters:
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from column Revenue, table_id file_0_view_0)
- $d_i$: Demand for product $i$ (from column Demand, table_id file_0_view_0)
- $I_i$: Initial Inventory for product $i$ (from column Initial Inventory, table_id file_0_view_0)

Decision Variables:
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill; $x_i \in \mathbb{Z}_+$ (non-negative integers)

Objective:
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

Constraints:
\[
x_i \leq d_i \quad \forall i \in \mathcal{I}
\]
\[
x_i \leq I_i \quad \forall i \in \mathcal{I}
\]
\[
x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
\]

Data Mapping:
- Index set $\mathcal{I}$: All rows in table_id file_0_view_0 with Product_Name classified under ‘Books’
- Parameter $A_i$: Revenue from column Revenue, table_id file_0_view_0
- Parameter $d_i$: Demand from column Demand, table_id file_0_view_0
- Parameter $I_i$: Initial Inventory from column Initial Inventory, table_id file_0_view_0
- Variable $x_i$: Number of units fulfilled for product $i$ (Books), as defined above

No additional constraints or data sources are imposed by the query.