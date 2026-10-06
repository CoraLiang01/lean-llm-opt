Mathematical Optimization Model

Index Sets:
- \( I \): Set of all pizza types, indexed by \( i \). (Data: all unique values in "Product Name" from table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of pizza type \( i \). (Data: "Revenue", file_0_view_0)
- \( d_i \): Demand for pizza type \( i \). (Data: "Demand", file_0_view_0)
- \( s_i \): Initial inventory available for pizza type \( i \). (Data: "Initial Inventory", file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of pizza type \( i \) to fulfill (integer, \( x_i \geq 0 \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \in \mathbb{Z}_+ && \forall i \in I \quad \text{(non-negative integers)}
\end{align*}
\]

Data Mapping:
- \( I \): All "Product Name" entries from table_id file_0_view_0
- \( r_i \): "Revenue" column, table_id file_0_view_0
- \( d_i \): "Demand" column, table_id file_0_view_0
- \( s_i \): "Initial Inventory" column, table_id file_0_view_0