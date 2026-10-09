Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- Let \( I \) be the set of all products in table_id file_0_view_0 whose 'Product Name' contains '27in'.

Parameters:
- \( r_i \): Revenue per unit of product \( i \), from column 'Revenue' in file_0_view_0.
- \( d_i \): Demand for product \( i \), from column 'Demand' in file_0_view_0.
- \( s_i \): Initial inventory for product \( i \), from column 'Initial Inventory' in file_0_view_0.

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of all '27in' products.)

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0 \text{ and integer} && \forall i \in I
\end{align*}
\]

Data Mapping:
- Index set \( I \): All rows in table_id file_0_view_0 where 'Product Name' contains '27in'.
- Parameter \( r_i \): 'Revenue' column, table_id file_0_view_0.
- Parameter \( d_i \): 'Demand' column, table_id file_0_view_0.
- Parameter \( s_i \): 'Initial Inventory' column, table_id file_0_view_0.
- Decision variable \( x_i \): Number of units to fulfill for each \( i \in I \).

This model maximizes total revenue from '27in' products, subject to demand and inventory limits for each product.