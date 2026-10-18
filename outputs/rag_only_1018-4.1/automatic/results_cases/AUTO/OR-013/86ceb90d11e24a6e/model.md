ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all “4U” products, indexed by \( i \).
 (Data Mapping: "Product Name" from table_id: file_0_view_0, filtered where "Product Name" starts with "4U")

Parameters:
- \( r_i \): Revenue per unit for product \( i \).
 (Data Mapping: "Revenue" from table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \) during the sales horizon.
 (Data Mapping: "Demand" from table_id: file_0_view_0)
- \( s_i \): Initial inventory for product \( i \).
 (Data Mapping: "Initial Inventory" from table_id: file_0_view_0)

Decision Variables:
- \( x_i \in \mathbb{Z}_+ \): Number of units of product \( i \) to fulfill (non-negative integer).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled sales of “4U” products.)

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Cannot fulfill more than available inventory)} \\
& x_i \geq 0,\ x_i \in \mathbb{Z} && \forall i \in I \quad \text{(Non-negative integer fulfillment)}
\end{align*}
\]

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( d_i \), \( s_i \) are sourced from table_id: file_0_view_0 ("OnlineSalesinUSA.csv"), using columns:
 - "Product Name" (filtered: "Product Name" starts with "4U")
 - "Revenue"
 - "Demand"
 - "Initial Inventory"

No additional constraints, relaxations, or modifications are imposed beyond those specified in the user query and the returned data.