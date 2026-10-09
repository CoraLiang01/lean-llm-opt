#### Mathematical Optimization Model

Let:
- $I$ = index set of all products with "Product Name" starting with "TABLET" (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue")
    - $d_i$ = demand for product $i$ (parameter from column "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: file_0_view_0 (from SmartphoneRetailOutletSalesData.csv)
    - Index set $I$: All rows where "Product Name" starts with "TABLET"
    - $A_i$: "Revenue" column
    - $d_i$: "Demand" column
    - $s_i$: "Initial Inventory" column

No additional constraints or synthetic capacities are specified in the query. All parameters and index sets are defined directly from the returned data.