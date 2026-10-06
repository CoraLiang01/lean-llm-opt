ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products i classified as ‘FAUX’ (from ZARASales.csv, see Data Mapping).

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column Revenue, table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) (from column Demand, table_id: file_0_view_0).
- \( s_i \): Initial inventory for product \( i \) (from column Initial Inventory, table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers).

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

3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All rows in ZARASales.csv (table_id: file_0_view_0) where Product Name contains or starts with ‘FAUX’.
- Parameter \( r_i \): ZARASales.csv, column Revenue, table_id: file_0_view_0.
- Parameter \( d_i \): ZARASales.csv, column Demand, table_id: file_0_view_0.
- Parameter \( s_i \): ZARASales.csv, column Initial Inventory, table_id: file_0_view_0.

No literal record values or counts are included; all data is referenced symbolically.