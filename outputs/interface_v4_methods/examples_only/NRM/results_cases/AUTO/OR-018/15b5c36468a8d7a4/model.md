ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products classified under 'Baby'. (From Salesdata.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (Salesdata.csv, file_0_view_0, column: Revenue)
- \( d_i \): Demand for product \( i \). (Salesdata.csv, file_0_view_0, column: Demand)
- \( s_i \): Initial inventory of product \( i \). (Salesdata.csv, file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer, for all \( i \in I \).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units of 'Baby' products.)

Constraints:
1. Inventory constraint: \( x_i \leq s_i \), for all \( i \in I \).
2. Demand constraint: \( x_i \leq d_i \), for all \( i \in I \).
3. Non-negativity and integrality: \( x_i \geq 0 \), integer, for all \( i \in I \).

Data Mapping:
- Index set \( I \): All rows in Salesdata.csv (table_id: file_0_view_0) where Product Name starts with "Baby".
- Parameter \( r_i \): Salesdata.csv, file_0_view_0, column: Revenue.
- Parameter \( d_i \): Salesdata.csv, file_0_view_0, column: Demand.
- Parameter \( s_i \): Salesdata.csv, file_0_view_0, column: Initial Inventory.