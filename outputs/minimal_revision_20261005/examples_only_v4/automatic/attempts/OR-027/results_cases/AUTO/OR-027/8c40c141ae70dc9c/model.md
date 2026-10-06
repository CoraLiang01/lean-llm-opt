Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘Organ’, indexed by \( i \). (From all rows in file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for product \( i \). (file_0_view_0, column: Demand)
- \( s_i \): Initial inventory of product \( i \). (file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity)} \\
& x_i \in \mathbb{Z} && \forall i \in I \quad \text{(integer units)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in table_id file_0_view_0 (column: Sub Category contains ‘Organ’)
- \( r_i \): file_0_view_0, column: Revenue
- \( d_i \): file_0_view_0, column: Demand
- \( s_i \): file_0_view_0, column: Initial Inventory