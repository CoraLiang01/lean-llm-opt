ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products classified under ‘ELE-S’, indexed by \( i \).

Parameters:
- \( r_i \): Revenue per unit of product \( i \). (Source: SalesStoreoverview.csv, column ‘Revenue’)
- \( s_i \): Initial inventory of product \( i \). (Source: SalesStoreoverview.csv, column ‘Initial Inventory’)
- \( d_i \): Demand quantity for product \( i \). (Source: SalesStoreoverview.csv, column ‘Demand’)

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
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All rows in SalesStoreoverview.csv where ‘Product_Reference’ starts with ‘ELE-S’.
- \( r_i \): SalesStoreoverview.csv, column ‘Revenue’, table_id: file_0_view_0
- \( s_i \): SalesStoreoverview.csv, column ‘Initial Inventory’, table_id: file_0_view_0
- \( d_i \): SalesStoreoverview.csv, column ‘Demand’, table_id: file_0_view_0

This model maximizes total revenue from fulfilling demand for ‘ELE-S’ products, subject to inventory and demand limits.