Abstract Optimization Model

Index Sets:
- \( I \): Set of all products classified under 'Baby' (from Salesdata.csv, Product Name with prefix 'Baby').

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (Salesdata.csv, Revenue, table_id: file_0_view_0, column: Revenue).
- \( d_i \): Demand for product \( i \) (Salesdata.csv, Demand, table_id: file_0_view_0, column: Demand).
- \( s_i \): Initial inventory for product \( i \) (Salesdata.csv, Initial Inventory, table_id: file_0_view_0, column: Initial Inventory).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), \( \forall i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]

Constraints:
1. Inventory and demand fulfillment bounds:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All rows in Salesdata.csv (table_id: file_0_view_0) where Product Name has prefix 'Baby'.
- Parameter \( r_i \): Salesdata.csv, table_id: file_0_view_0, column: Revenue.
- Parameter \( d_i \): Salesdata.csv, table_id: file_0_view_0, column: Demand.
- Parameter \( s_i \): Salesdata.csv, table_id: file_0_view_0, column: Initial Inventory.