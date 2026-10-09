#### Mathematical Optimization Model

Let:
- $I$ = set of all products with $id\_number$ prefix "id999" (from the data, $I = \{\text{id999}\}$).
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter from column "Revenue").
    - $d_i$ = demand for product $i$ during the sales horizon (parameter from column "Demand").
    - $s_i$ = initial inventory of product $i$ (parameter from column "Initial Inventory").
    - $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$).

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: OnlineRetailSalesDataset.csv (table_id: file_0_view_0)
    - Index set $I$: All rows where "id_number" has prefix "id999"
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $s_i$: column "Initial Inventory"