Index Sets:
- Let \( I \) be the set of products classified under ‘27in’, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit for product \( i \) (from ‘Revenue’ column).
- \( s_i \): Initial inventory for product \( i \) (from ‘Initial Inventory’ column).
- \( d_i \): Demand quantity for product \( i \) (from ‘Demand’ column).

Decision Variables:
- \( x_i \geq 0 \): Number of units of product \( i \) to fulfill.

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq s_i && \forall i \in I \quad \text{(cannot exceed initial inventory)} \\
& x_i \leq d_i && \forall i \in I \quad \text{(cannot exceed demand)} \\
& x_i \geq 0 && \forall i \in I \quad \text{(non-negativity)}
\end{align*}
\]

Data Mapping:
- Index set \( I \): All records in table_id file_0_view_0 where ‘Product Name’ has prefix ‘27in’ (CSVQA filter: {"column": "Product Name", "operator": "prefix", "value": "27in"}).
- \( r_i \): ‘Revenue’ column in table_id file_0_view_0.
- \( s_i \): ‘Initial Inventory’ column in table_id file_0_view_0.
- \( d_i \): ‘Demand’ column in table_id file_0_view_0.