#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with names starting with "27in" (from column "Product Name" in table_id file_0_view_0)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column "Revenue")
    - $d_i$ = total demand for product $i$ (from column "Demand")
    - $s_i$ = initial inventory of product $i$ (from column "Initial Inventory")
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

Objective:
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \geq 0, \quad \forall i \in I \\
& x_i \in \mathbb{Z}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Index set $I$: All rows in table_id file_0_view_0 where "Product Name" starts with "27in"
- Parameter $A_i$: "Revenue" column in table_id file_0_view_0
- Parameter $d_i$: "Demand" column in table_id file_0_view_0
- Parameter $s_i$: "Initial Inventory" column in table_id file_0_view_0
- Variable $x_i$: Number of units fulfilled for each $i \in I$ (decision variable)