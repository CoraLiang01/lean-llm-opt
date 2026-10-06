Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products classified under ‘FAUX’  
  (Data: all "Product Name" in table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \)  
  (Data: "Revenue" in table_id file_0_view_0, for each \( i \in I \))
- \( d_i \): Demand quantity for product \( i \)  
  (Data: "Demand" in table_id file_0_view_0, for each \( i \in I \))
- \( s_i \): Initial inventory for product \( i \)  
  (Data: "Initial Inventory" in table_id file_0_view_0, for each \( i \in I \))

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill  
  Domain: \( 0 \leq x_i \leq \min\{d_i, s_i\} \), integer, for all \( i \in I \)

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units)

Constraints:
1. Inventory and demand fulfillment:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
\]
(Do not fulfill more than available inventory or demand for any product)

Data Mapping:
- Index set \( I \): All "Product Name" in table_id file_0_view_0
- \( r_i \): "Revenue" in table_id file_0_view_0
- \( d_i \): "Demand" in table_id file_0_view_0
- \( s_i \): "Initial Inventory" in table_id file_0_view_0

No additional constraints or data sources are used.