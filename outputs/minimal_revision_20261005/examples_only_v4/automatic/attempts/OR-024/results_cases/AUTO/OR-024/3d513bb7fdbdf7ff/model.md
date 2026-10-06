Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘S700_’, indexed by \( i \).
  (Source: SampleSalesData.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Source: file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for product \( i \).
  (Source: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory of product \( i \).
  (Source: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
\[
\begin{align*}
& 0 \leq x_i \leq d_i, && \forall i \in I \\
& 0 \leq x_i \leq s_i, && \forall i \in I \\
& x_i \in \mathbb{Z}, && \forall i \in I
\end{align*}
\]
(Do not fulfill more than demand or available inventory for each product.)

Data Mapping

- Index set \( I \): All records in file_0_view_0, column Product Name (filtered for prefix ‘S700_’)
- Parameter \( r_i \): file_0_view_0, column Revenue
- Parameter \( d_i \): file_0_view_0, column Demand
- Parameter \( s_i \): file_0_view_0, column Initial Inventory