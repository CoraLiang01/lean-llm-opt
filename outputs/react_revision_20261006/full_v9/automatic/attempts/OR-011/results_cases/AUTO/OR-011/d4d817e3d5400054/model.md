#### Mathematical Optimization Model

Let:
- $I$ = set of all products with $id\_number$ prefix "id999" (from the data, all such products indexed by $i \in I$)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$
    - $d_i$ = demand for product $i$ over the sales horizon
    - $s_i$ = initial inventory of product $i$
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
\[
\begin{align*}
& x_i \leq d_i, \quad \forall i \in I \\
& x_i \leq s_i, \quad \forall i \in I \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

#### Data Mapping

- Table: OnlineRetailSalesDataset.csv
    - Index set $I$: All rows where column "id_number" has prefix "id999" (table_id: file_0_view_0, column: id_number)
    - Parameter $A_i$: Revenue per unit from column "Revenue" (table_id: file_0_view_0, column: Revenue)
    - Parameter $d_i$: Demand from column "Demand" (table_id: file_0_view_0, column: Demand)
    - Parameter $s_i$: Initial Inventory from column "Initial Inventory" (table_id: file_0_view_0, column: Initial Inventory)