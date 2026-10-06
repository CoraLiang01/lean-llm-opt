Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘id999’. (Data: file_0_view_0, column: id_number)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (Data: file_0_view_0, column: Revenue)
- \( d_i \): Demand for product \( i \) during the sales horizon. (Data: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory of product \( i \). (Data: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+, \forall i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(Cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(Cannot fulfill more than available inventory)} \\
& x_i \geq 0,\ x_i \in \mathbb{Z} && \forall i \in I \quad \text{(Non-negative integer fulfillment)}
\end{align*}
\]

Data Mapping

- Index set \( I \): file_0_view_0, column id_number (filtered for prefix ‘id999’)
- Parameter \( r_i \): file_0_view_0, column Revenue
- Parameter \( d_i \): file_0_view_0, column Demand
- Parameter \( s_i \): file_0_view_0, column Initial Inventory