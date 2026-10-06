ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products classified under 'ELE-S', indexed by \( i \). (From SalesStoreoverview.csv, table_id: file_0_view_0, column: Product_Reference)

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (table_id: file_0_view_0, column: Revenue)
- \( s_i \): Initial inventory of product \( i \). (table_id: file_0_view_0, column: Initial Inventory)
- \( d_i \): Demand for product \( i \). (table_id: file_0_view_0, column: Demand)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (if required by context).

Objective:
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
   (Cannot fulfill more than available inventory.)

2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   (Cannot fulfill more than demand.)

3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]
   (Cannot fulfill negative units.)

Data Mapping:
- Index set \( I \): All rows in SalesStoreoverview.csv (table_id: file_0_view_0) where Product_Reference starts with 'ELE-S'.
- Parameter \( r_i \): Revenue column in SalesStoreoverview.csv (table_id: file_0_view_0).
- Parameter \( s_i \): Initial Inventory column in SalesStoreoverview.csv (table_id: file_0_view_0).
- Parameter \( d_i \): Demand column in SalesStoreoverview.csv (table_id: file_0_view_0).

No record values or literal record counts are included. All data sources are explicitly mapped.