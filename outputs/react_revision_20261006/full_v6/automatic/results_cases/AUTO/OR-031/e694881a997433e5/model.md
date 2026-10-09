#### Mathematical Optimization Model

Let:
- $I$ = set of all dairy products (indexed by $i$), as defined by the Full_Product_Name column.
- $A_i$ = revenue per unit of product $i$ (parameter from Revenue column, table_id: file_0_view_0).
- $d_i$ = deterministic demand for product $i$ (parameter from Demand column, table_id: file_0_view_0).
- $s_i$ = initial inventory for product $i$ (parameter from Initial Inventory column, table_id: file_0_view_0).
- $x_i$ = number of units of product $i$ to fulfill (decision variable, integer, $x_i \geq 0$).

Objective:
\[
\max \sum_{i \in I} A_i x_i
\]

Subject to:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \\
& x_i \leq s_i && \forall i \in I \\
& x_i \in \mathbb{Z}_+, && \forall i \in I
\end{align*}
\]

#### Data Mapping

- Index set $I$: All unique values in Full_Product_Name from table_id: file_0_view_0, column: Full_Product_Name.
- Parameter $A_i$: Revenue per unit from table_id: file_0_view_0, column: Revenue.
- Parameter $d_i$: Demand from table_id: file_0_view_0, column: Demand.
- Parameter $s_i$: Initial Inventory from table_id: file_0_view_0, column: Initial Inventory.
- Decision variable $x_i$: Number of units fulfilled for product $i$.

All parameters are mapped directly from the specified columns in DairyGoodsSalesDataset.csv (table_id: file_0_view_0).