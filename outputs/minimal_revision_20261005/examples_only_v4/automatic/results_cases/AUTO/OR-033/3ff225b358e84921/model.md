Mathematical Optimization Model

Sets:
- \( I \): Set of all products classified under ‘Baby’  
  (Data: all "Product Name" in table_id file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \)  
  (Data: "Revenue" in table_id file_0_view_0, indexed by \( i \in I \))
- \( d_i \): Demand quantity for product \( i \)  
  (Data: "Demand" in table_id file_0_view_0, indexed by \( i \in I \))
- \( s_i \): Initial inventory for product \( i \)  
  (Data: "Initial Inventory" in table_id file_0_view_0, indexed by \( i \in I \))

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill  
  (domain: integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \))

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
   (Cannot fulfill more than available inventory or demand for each product)

Data Mapping:
- Set \( I \): All "Product Name" in table_id file_0_view_0 (filtered for prefix "Baby")
- Parameter \( r_i \): "Revenue" in table_id file_0_view_0
- Parameter \( d_i \): "Demand" in table_id file_0_view_0
- Parameter \( s_i \): "Initial Inventory" in table_id file_0_view_0

No additional constraints or data sources are present.