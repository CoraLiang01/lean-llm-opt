#### Mathematical Optimization Model

Let:
- $I$ = index set of all products with names starting with "27in" (from the source data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter, from column "Revenue")
    - $d_i$ = total demand for product $i$ (parameter, from column "Demand")
    - $s_i$ = initial inventory for product $i$ (parameter, from column "Initial Inventory")
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

- Table: file_0_view_0 (from Salesorders.csv)
    - Index set $I$: all rows where "Product Name" starts with "27in"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"
    - Decision variable $x_i$: number of units of product $i$ to fulfill

No additional constraints or synthetic scenario parameters are specified in the query. All bounds and parameters are mapped directly from the source data.