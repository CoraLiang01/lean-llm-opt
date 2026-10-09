#### Symbolic Mathematical Model

Let:
- $I$ = set of all “4U” products (from the source data, all products whose "Product Name" starts with "4U")
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from column "Revenue")
    - $d_i$ = total demand for product $i$ over the sales horizon (parameter, from column "Demand")
    - $s_i$ = initial inventory of product $i$ (parameter, from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
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

- Table: OnlineSalesinUSA.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where "Product Name" starts with "4U"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"