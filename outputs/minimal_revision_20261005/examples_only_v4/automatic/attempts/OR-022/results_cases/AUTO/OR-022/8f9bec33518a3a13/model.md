Mathematical Optimization Model

Sets:
- Let \( I \) be the set of all products in table_id file_0_view_0 whose Product Name contains '27in'.

Parameters (for each \( i \in I \)):
- \( r_i \): Revenue per unit of product \( i \) (from column Revenue)
- \( d_i \): Demand for product \( i \) (from column Demand)
- \( s_i \): Initial inventory of product \( i \) (from column Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity and integrality)}
\end{align*}
\]

Data Mapping

- Set \( I \): All records in table_id file_0_view_0 where Product Name contains '27in'
- Parameter \( r_i \): file_0_view_0, column Revenue
- Parameter \( d_i \): file_0_view_0, column Demand
- Parameter \( s_i \): file_0_view_0, column Initial Inventory