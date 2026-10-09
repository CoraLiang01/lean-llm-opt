Sets and Indices:
- Let \( I \) be the set of all “4U” products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit for product \( i \) (from column Revenue, table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) during the sales horizon (from column Demand, table_id: file_0_view_0).
- \( s_i \): Initial inventory for product \( i \) (from column Initial Inventory, table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \in \mathbb{Z}_{\geq 0} && \forall i \in I \\
\end{align*}
\]

Data Mapping:
- Set \( I \): All records in table_id: file_0_view_0 where Product Name has prefix "4U" (CSVQA-applied filter).
- \( r_i \): Revenue, table_id: file_0_view_0, column: Revenue.
- \( d_i \): Demand, table_id: file_0_view_0, column: Demand.
- \( s_i \): Initial Inventory, table_id: file_0_view_0, column: Initial Inventory.

This model maximizes total revenue from “4U” products by optimally allocating available inventory to meet deterministic demand, subject to inventory and demand limits, with all data and subset selection as returned by CSVQA.