Sets:
- \( I \): Index set of products classified under ‘Baby’. (From Salesdata.csv, table_id: file_0_view_0, where Product Name has prefix 'Baby')

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column 'Revenue', table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \) (from column 'Demand', table_id: file_0_view_0)
- \( s_i \): Initial inventory for product \( i \) (from column 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
\[
\begin{align*}
& x_i \leq d_i && \forall i \in I \\
& x_i \leq s_i && \forall i \in I \\
& x_i \geq 0 && \forall i \in I \\
& x_i \in \mathbb{Z} && \forall i \in I
\end{align*}
\]

Data Mapping:
- Set \( I \), and parameters \( r_i \), \( d_i \), \( s_i \) are sourced from Salesdata.csv (table_id: file_0_view_0), using only records where 'Product Name' has prefix 'Baby'. Specifically:
    - \( r_i \): 'Revenue' column
    - \( d_i \): 'Demand' column
    - \( s_i \): 'Initial Inventory' column

No restocking is allowed; all constraints and data are based solely on the initial inventory and demand for each ‘Baby’ product.