Abstract Mathematical Optimization Model

Index Sets:
- \( I \): Set of products, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (from 'Revenue')
- \( d_i \): Demand for product \( i \). (from 'Demand')
- \( s_i \): Initial inventory available for product \( i \). (from 'Initial Inventory')

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity, integer)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in table_id "file_0_view_0", column "Product Name".
- Parameter \( r_i \): "Revenue" column in table_id "file_0_view_0".
- Parameter \( d_i \): "Demand" column in table_id "file_0_view_0".
- Parameter \( s_i \): "Initial Inventory" column in table_id "file_0_view_0".
- Decision variable \( x_i \): Number of units to fulfill for each product \( i \in I \).

All data is sourced directly from "file_0_view_0" with no additional filters.