ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of all products in the 'Organ' category (indexed by \( i \)), where each \( i \) corresponds to a unique value in Sub Category from table_id file_0_view_0.

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column Revenue, table_id file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) (from column Initial Inventory, table_id file_0_view_0).
- \( d_i \): Demand for product \( i \) (from column Demand, table_id file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer or continuous as appropriate.

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
  \]

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
- Index set \( I \): All rows in file_0_view_0 where Sub Category contains 'Organ'.
- Parameter \( r_i \): Revenue column, table_id file_0_view_0.
- Parameter \( s_i \): Initial Inventory column, table_id file_0_view_0.
- Parameter \( d_i \): Demand column, table_id file_0_view_0.

This model maximizes total revenue from fulfilling demand for 'Organ' products, subject to inventory and demand limits, using the provided data.