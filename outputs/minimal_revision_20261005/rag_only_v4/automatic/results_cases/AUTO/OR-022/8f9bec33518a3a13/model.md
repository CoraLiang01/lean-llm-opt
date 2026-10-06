Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products in file_0_view_0 whose 'Product Name' contains '27in'.

Parameters:
- \( r_i \): Revenue per unit of product \( i \), from 'Revenue' (file_0_view_0).
- \( d_i \): Demand for product \( i \), from 'Demand' (file_0_view_0).
- \( s_i \): Initial inventory of product \( i \), from 'Initial Inventory' (file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]

Variable Domains:
- \( x_i \) are integer variables.

Data Mapping

- Index set \( I \): All records in table_id file_0_view_0 where 'Product Name' contains '27in'.
- \( r_i \): file_0_view_0, column 'Revenue'.
- \( d_i \): file_0_view_0, column 'Demand'.
- \( s_i \): file_0_view_0, column 'Initial Inventory'.