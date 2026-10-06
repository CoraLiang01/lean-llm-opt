#### Symbolic Mathematical Model

Let:

- $I$ = index set of products classified under ‘id999’ (from table_id: file_0_view_0, column: id_number)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (from table_id: file_0_view_0, column: Revenue)
    - $d_i$ = demand for product $i$ during the sales horizon (from table_id: file_0_view_0, column: Demand)
    - $s_i$ = initial inventory of product $i$ (from table_id: file_0_view_0, column: Initial Inventory)
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
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Index set $I$: All rows in table_id: file_0_view_0 where column id_number = 'id999'
- Parameter $A_i$: table_id: file_0_view_0, column: Revenue
- Parameter $d_i$: table_id: file_0_view_0, column: Demand
- Parameter $s_i$: table_id: file_0_view_0, column: Initial Inventory
- Variable $x_i$: Decision variable for each $i \in I$