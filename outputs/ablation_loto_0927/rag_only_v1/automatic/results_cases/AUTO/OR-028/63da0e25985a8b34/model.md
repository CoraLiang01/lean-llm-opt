ABSTRACT OPTIMIZATION MODEL

Index Sets:
- \( I \): Set of products (from table_id: file_0_view_0, column: Product Name)

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from table_id: file_0_view_0, column: Revenue)
- \( d_i \): Demand for product \( i \) (from table_id: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for product \( i \) (from table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \geq 0 \), integer

Objective:
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
- Index set \( I \): file_0_view_0, column 'Product Name'
- Parameter \( r_i \): file_0_view_0, column 'Revenue'
- Parameter \( d_i \): file_0_view_0, column 'Demand'
- Parameter \( s_i \): file_0_view_0, column 'Initial Inventory'