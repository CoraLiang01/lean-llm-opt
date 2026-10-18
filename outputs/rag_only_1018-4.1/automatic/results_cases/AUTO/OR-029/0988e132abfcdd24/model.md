Abstract Mathematical Optimization Model

Index Sets:
- \( I \): Set of products where 'Product Name' starts with 'FAUX' (from ZARASales.csv, table_id: file_0_view_0).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from 'Revenue', table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) (from 'Demand', table_id: file_0_view_0).
- \( s_i \): Initial inventory for product \( i \) (from 'Initial Inventory', table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( x_i \geq 0 \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory and demand fulfillment bounds for each product:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]
   (i.e., for each product, the fulfilled quantity cannot exceed either its demand or its initial inventory.)

Variable Domains:
- \( x_i \) are integer variables, \( x_i \geq 0 \).

Data Mapping:
- Index set \( I \), and parameters \( r_i \), \( d_i \), \( s_i \) are sourced from ZARASales.csv (table_id: file_0_view_0), using only rows where 'Product Name' has prefix 'FAUX' (CSVQA-applied filter: 'Product Name' prefix 'FAUX').
- \( r_i \): 'Revenue' column, table_id: file_0_view_0
- \( d_i \): 'Demand' column, table_id: file_0_view_0
- \( s_i \): 'Initial Inventory' column, table_id: file_0_view_0

No additional constraints or requirements are imposed beyond those specified in the user query and the data returned by CSVQA.