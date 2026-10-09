Sets and Indices:
- Let \( I \) be the set of all products whose "Product Name" begins with "4U" (i.e., the set of “4U” products), indexed by \( i \in I \).

Parameters:
- \( r_i \): Revenue per unit for product \( i \). (From column "Revenue")
- \( d_i \): Demand for product \( i \) during the sales horizon. (From column "Demand")
- \( s_i \): Initial inventory available for product \( i \). (From column "Initial Inventory")

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, where \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

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
& x_i \geq 0 \text{ and integer} && \forall i \in I \\
\end{align*}
\]

Data Mapping:
- Source table: OnlineSalesinUSA.csv (table_id: file_0_view_0)
- Index set \( I \): All records where "Product Name" has prefix "4U" (i.e., “Product Name” column, filter: prefix = "4U")
- \( r_i \): "Revenue" column, for each \( i \in I \)
- \( d_i \): "Demand" column, for each \( i \in I \)
- \( s_i \): "Initial Inventory" column, for each \( i \in I \)

This model maximizes total revenue from “4U” products by choosing fulfillment quantities that do not exceed either available inventory or realized demand for each product. No restocking or in-transit inventory is allowed.