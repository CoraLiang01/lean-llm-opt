Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- \( I \): Set of all “4U” products, indexed by \( i \). (From all rows where "Product Name" has prefix "4U" in table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (file_0_view_0, column "Revenue")
- \( d_i \): Demand for product \( i \) during the sales horizon. (file_0_view_0, column "Demand")
- \( s_i \): Initial inventory of product \( i \). (file_0_view_0, column "Initial Inventory")

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

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
- Index set \( I \): All products in file_0_view_0 where "Product Name" starts with "4U".
- Parameter \( r_i \): file_0_view_0, column "Revenue".
- Parameter \( d_i \): file_0_view_0, column "Demand".
- Parameter \( s_i \): file_0_view_0, column "Initial Inventory".