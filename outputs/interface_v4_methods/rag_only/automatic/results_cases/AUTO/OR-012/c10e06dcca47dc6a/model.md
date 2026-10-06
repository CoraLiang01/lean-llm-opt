Abstract Mathematical Optimization Model

Sets:
- \( I \): Set of products (indexed by \( i \)), corresponding to "Product Name" in table_id: file_0_view_0.

Parameters:
- \( r_i \): Per-unit revenue for product \( i \) ("Revenue", file_0_view_0).
- \( d_i \): Deterministic demand for product \( i \) ("Demand", file_0_view_0).
- \( s_i \): Initial inventory for product \( i \) ("Initial Inventory", file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill for customer purchases (\( x_i \geq 0 \), integer or continuous as appropriate).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than available inventory)} \\
& x_i \geq 0 && \forall i \in I
\end{align*}
\]

Data Mapping:
- Set \( I \): "Product Name" in table_id: file_0_view_0
- Parameter \( r_i \): "Revenue" in table_id: file_0_view_0
- Parameter \( d_i \): "Demand" in table_id: file_0_view_0
- Parameter \( s_i \): "Initial Inventory" in table_id: file_0_view_0

This model maximizes total revenue by optimally allocating inventory to meet deterministic demand, subject to inventory and demand limits for each product.