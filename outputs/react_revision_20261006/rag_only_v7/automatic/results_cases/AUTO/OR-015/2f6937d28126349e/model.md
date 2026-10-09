Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘Aalop’. (From all "Product Name" entries in table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (From "Revenue" in file_0_view_0)
- \( d_i \): Demand for product \( i \) during the sales horizon. (From "Demand" in file_0_view_0)
- \( s_i \): Initial inventory of product \( i \). (From "Initial Inventory" in file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), \( x_i \in \mathbb{Z}_+ \), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \quad \text{(cannot fulfill more than demand)} \\
& x_i \leq s_i && \forall i \in I \quad \text{(cannot fulfill more than initial inventory)} \\
& x_i \geq 0,\ x_i \in \mathbb{Z} && \forall i \in I \\
\end{align*}
\]

Data Mapping:
- Index set \( I \): All "Product Name" in table_id file_0_view_0, filtered where "Product Name" has prefix "Aalop".
- Parameter \( r_i \): "Revenue" column in file_0_view_0.
- Parameter \( d_i \): "Demand" column in file_0_view_0.
- Parameter \( s_i \): "Initial Inventory" column in file_0_view_0.