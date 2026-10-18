Abstract Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘Aalop’. (From RestaurantSalesreport.csv, table_id: file_0_view_0, filtered where Product Name has prefix 'Aalop')

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column 'Revenue', table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \) during the sales horizon (from column 'Demand', table_id: file_0_view_0)
- \( s_i \): Initial inventory available for product \( i \) (from column 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer, for all \( i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of ‘Aalop’ products.)

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
   (Cannot fulfill more than initial inventory.)

2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   (Cannot fulfill more than demand.)

3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All records in RestaurantSalesreport.csv (table_id: file_0_view_0) where Product Name has prefix 'Aalop' (CSVQA-applied filter).
- Parameter \( r_i \): 'Revenue' column, table_id: file_0_view_0.
- Parameter \( d_i \): 'Demand' column, table_id: file_0_view_0.
- Parameter \( s_i \): 'Initial Inventory' column, table_id: file_0_view_0.

All data and constraints are based solely on the CSVQA-applied filter and columns. No additional products or constraints are included.