Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products in file_0_view_0 whose 'Product Name' contains '27in'.

Parameters:
- \( r_i \): Revenue per unit of product \( i \), from 'Revenue' (file_0_view_0).
- \( d_i \): Demand for product \( i \), from 'Demand' (file_0_view_0).
- \( s_i \): Initial inventory for product \( i \), from 'Initial Inventory' (file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity, integer)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in file_0_view_0 where 'Product Name' contains '27in'.
- Parameter \( r_i \): file_0_view_0, column 'Revenue'.
- Parameter \( d_i \): file_0_view_0, column 'Demand'.
- Parameter \( s_i \): file_0_view_0, column 'Initial Inventory'.