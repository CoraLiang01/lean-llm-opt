#### Symbolic Optimization Model

Let:

- $I$ = index set of all products with Product_Reference starting with ‘ELE-S’ (from table_id: file_0_view_0, column: Product_Reference)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from column: Revenue)
    - $d_i$ = demand for product $i$ (from column: Demand)
    - $s_i$ = initial inventory of product $i$ (from column: Initial Inventory)
    - $x_i$ = integer decision variable: number of units of product $i$ to fulfill

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

- Index set $I$ and product identifiers: table_id: file_0_view_0, column: Product_Reference (filtered by prefix ‘ELE-S’)
- Parameter $A_i$: table_id: file_0_view_0, column: Revenue
- Parameter $d_i$: table_id: file_0_view_0, column: Demand
- Parameter $s_i$: table_id: file_0_view_0, column: Initial Inventory