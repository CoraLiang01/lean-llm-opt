Mathematical Optimization Model

Index Sets:
- \( I \): Set of all pizza types, indexed by \( i \). (Data: all unique values in "Product Name" from table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of pizza type \( i \). (Data: "Revenue", file_0_view_0)
- \( d_i \): Demand for pizza type \( i \). (Data: "Demand", file_0_view_0)
- \( s_i \): Initial inventory available for pizza type \( i \). (Data: "Initial Inventory", file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of pizza type \( i \) to fulfill (integer, \( x_i \geq 0 \), \( x_i \in \mathbb{Z}_+ \))

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \geq 0,\ x_i \in \mathbb{Z} && \forall i \in I \quad \text{(non-negative integer)}
\end{align*}
\]

Data Mapping:
- \( I \): All "Product Name" in table_id file_0_view_0
- \( r_i \): "Revenue" in table_id file_0_view_0, mapped by "Product Name"
- \( d_i \): "Demand" in table_id file_0_view_0, mapped by "Product Name"
- \( s_i \): "Initial Inventory" in table_id file_0_view_0, mapped by "Product Name"