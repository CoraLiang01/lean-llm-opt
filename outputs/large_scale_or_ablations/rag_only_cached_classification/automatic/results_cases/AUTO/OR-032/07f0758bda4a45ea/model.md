ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products classified as 'Books'. Each product \( i \in I \) corresponds to a row where Product_Name starts with 'Books' in table_id file_0_view_0.

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column 'Revenue', table_id file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) (from column 'Initial Inventory', table_id file_0_view_0).
- \( d_i \): Demand for product \( i \) (from column 'Demand', table_id file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (for all \( i \in I \)).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of 'Books' products.)

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All rows in table_id file_0_view_0 where Product_Name starts with 'Books'.
- Parameter \( r_i \): 'Revenue' column, table_id file_0_view_0.
- Parameter \( s_i \): 'Initial Inventory' column, table_id file_0_view_0.
- Parameter \( d_i \): 'Demand' column, table_id file_0_view_0.

No data values or record counts are included; all references are symbolic and mapped to their exact table_id and column names.