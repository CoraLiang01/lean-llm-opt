Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- \( I \): Set of all products classified under ‘Baby’ in the data. (From table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (file_0_view_0, Revenue)
- \( d_i \): Demand quantity for product \( i \). (file_0_view_0, Demand)
- \( s_i \): Initial inventory of product \( i \). (file_0_view_0, Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), \( \forall i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of ‘Baby’ products.)

Constraints:
1. Demand and Inventory Fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory or demand for each product.)

2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All records in table_id: file_0_view_0, column: Product Name, filtered to products classified under ‘Baby’.
- Parameter \( r_i \): file_0_view_0, column: Revenue.
- Parameter \( d_i \): file_0_view_0, column: Demand.
- Parameter \( s_i \): file_0_view_0, column: Initial Inventory.