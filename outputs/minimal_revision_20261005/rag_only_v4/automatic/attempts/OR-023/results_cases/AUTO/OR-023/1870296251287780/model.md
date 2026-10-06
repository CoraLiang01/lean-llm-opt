Mathematical Optimization Model

Index Sets:
- \( I \): Set of products classified under ‘ELE-S’, indexed by \( i \).
  (Data: Product_Reference in table_id file_0_view_0, filtered by prefix ‘ELE-S’)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data: Revenue, table_id file_0_view_0)
- \( d_i \): Demand quantity for product \( i \).
  (Data: Demand, table_id file_0_view_0)
- \( s_i \): Initial inventory for product \( i \).
  (Data: Initial Inventory, table_id file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Demand fulfillment and inventory limits:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
\]

Variable Domains:
- \( x_i \) are integer variables.

Data Mapping

- Index set \( I \): All Product_Reference in table_id file_0_view_0 with prefix ‘ELE-S’
- Parameter \( r_i \): Revenue, table_id file_0_view_0, column ‘Revenue’
- Parameter \( d_i \): Demand, table_id file_0_view_0, column ‘Demand’
- Parameter \( s_i \): Initial Inventory, table_id file_0_view_0, column ‘Initial Inventory’