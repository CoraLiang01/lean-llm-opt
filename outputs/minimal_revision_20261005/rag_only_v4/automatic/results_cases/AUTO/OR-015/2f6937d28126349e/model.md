Mathematical Optimization Model

Sets:
- \( I \): Set of products classified under ‘Aalop’, indexed by \( i \).
  (Data: all "Product Name" in table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \).
  (Data: "Revenue" in table_id file_0_view_0)
- \( d_i \): Demand for product \( i \) during the sales horizon.
  (Data: "Demand" in table_id file_0_view_0)
- \( s_i \): Initial inventory of product \( i \).
  (Data: "Initial Inventory" in table_id file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill.
  Domain: Integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of ‘Aalop’ products.)

Constraints:
1. Inventory and demand fulfillment:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
\]
 (Units fulfilled cannot exceed either demand or initial inventory.)

Data Mapping:
- Set \( I \): All "Product Name" in table_id file_0_view_0
- Parameter \( r_i \): "Revenue" in table_id file_0_view_0
- Parameter \( d_i \): "Demand" in table_id file_0_view_0
- Parameter \( s_i \): "Initial Inventory" in table_id file_0_view_0

No data is invented or omitted; all parameters and sets are bound to the specified columns in the provided table.