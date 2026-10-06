Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products classified under ‘ZZ’, indexed by \( i \).
  (Data: all SKUs in column SKU of table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data: column Revenue, table_id file_0_view_0)
- \( d_i \): Demand quantity for product \( i \).
  (Data: column Demand, table_id file_0_view_0)
- \( s_i \): Initial inventory for product \( i \).
  (Data: column Initial Inventory, table_id file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& 0 \leq x_i \leq d_i && \forall i \in I \\
& 0 \leq x_i \leq s_i && \forall i \in I \\
& x_i \in \mathbb{Z}_+ && \forall i \in I
\end{align*}
\]

Data Mapping

- Index set \( I \): All SKUs in column SKU of table_id file_0_view_0 (filtered to prefix ‘ZZ’)
- Parameter \( r_i \): Revenue from column Revenue, table_id file_0_view_0
- Parameter \( d_i \): Demand from column Demand, table_id file_0_view_0
- Parameter \( s_i \): Initial Inventory from column Initial Inventory, table_id file_0_view_0