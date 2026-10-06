Mathematical Optimization Model

Sets:
- Let \( I \) be the set of all products in file_0_view_0 such that 'Product Name' contains '27in'.

Parameters (for each \( i \in I \)):
- \( r_i \): Revenue per unit of product \( i \) (from 'Revenue', file_0_view_0)
- \( d_i \): Demand for product \( i \) (from 'Demand', file_0_view_0)
- \( s_i \): Initial inventory of product \( i \) (from 'Initial Inventory', file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Demand and Inventory Fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]

Variable Domains:
- \( x_i \) are integer variables.

Data Mapping

- Set \( I \): All records in table_id file_0_view_0 where 'Product Name' contains '27in'
- \( r_i \): file_0_view_0, column 'Revenue'
- \( d_i \): file_0_view_0, column 'Demand'
- \( s_i \): file_0_view_0, column 'Initial Inventory'