ABSTRACT Mathematical Optimization Model

Index Sets:
- \( I \): Set of all products classified as ‘FAUX’ (indexed by \( i \)), where each product is uniquely identified by its ‘Product Name’ from table_id: file_0_view_0.

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (‘Revenue’ column, table_id: file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) (‘Initial Inventory’ column, table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) (‘Demand’ column, table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer or continuous as appropriate.

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
- Index set \( I \): All rows in table_id: file_0_view_0 where ‘Product Name’ contains ‘FAUX’.
- Parameter \( r_i \): ‘Revenue’ column, table_id: file_0_view_0.
- Parameter \( s_i \): ‘Initial Inventory’ column, table_id: file_0_view_0.
- Parameter \( d_i \): ‘Demand’ column, table_id: file_0_view_0.

No literal record values or counts are included; all data references are symbolic and mapped to their exact table and column sources.