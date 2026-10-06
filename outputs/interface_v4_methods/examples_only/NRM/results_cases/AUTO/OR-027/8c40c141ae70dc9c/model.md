ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let \( I \) be the set of all products classified as ‘Organ’ (i.e., all products where Sub Category contains "Organ").

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from column ‘Revenue’ in table_id: file_0_view_0).
- \( s_i \): Initial inventory of product \( i \) (from column ‘Initial Inventory’ in table_id: file_0_view_0).
- \( d_i \): Demand for product \( i \) (from column ‘Demand’ in table_id: file_0_view_0).

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer (for all \( i \in I \)).

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
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set \( I \): All rows in table_id: file_0_view_0 where ‘Sub Category’ contains "Organ".
- Parameter \( r_i \): ‘Revenue’ column, table_id: file_0_view_0.
- Parameter \( s_i \): ‘Initial Inventory’ column, table_id: file_0_view_0.
- Parameter \( d_i \): ‘Demand’ column, table_id: file_0_view_0.