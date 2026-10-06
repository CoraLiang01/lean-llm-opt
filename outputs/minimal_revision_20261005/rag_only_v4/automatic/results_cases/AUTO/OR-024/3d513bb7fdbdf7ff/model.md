Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘S700_’. (Data: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \in I \). (Data: file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for product \( i \in I \). (Data: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for product \( i \in I \). (Data: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \in I \) to fulfill. (Domain: integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \))

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
\[
\begin{align*}
& 0 \leq x_i \leq d_i && \forall i \in I \quad \text{(Cannot fulfill more than demand)} \\
& 0 \leq x_i \leq s_i && \forall i \in I \quad \text{(Cannot fulfill more than inventory)} \\
\end{align*}
\]
(Equivalently, \( 0 \leq x_i \leq \min\{d_i, s_i\} \) for all \( i \in I \).)

Data Mapping:
- Index set \( I \): All products in file_0_view_0, column Product Name, where Product Name starts with ‘S700_’.
- Parameter \( r_i \): file_0_view_0, column Revenue.
- Parameter \( d_i \): file_0_view_0, column Demand.
- Parameter \( s_i \): file_0_view_0, column Initial Inventory.