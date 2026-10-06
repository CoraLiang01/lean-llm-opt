Abstract Optimization Model

Index Sets:
- \( I \): Set of products (from Product Name in table_id: file_0_view_0)

Parameters:
- \( r_i \): Revenue per unit of product \( i \) (from Revenue, table_id: file_0_view_0)
- \( d_i \): Demand for product \( i \) (from Demand, table_id: file_0_view_0)
- \( s_i \): Initial inventory for product \( i \) (from Initial Inventory, table_id: file_0_view_0)

Decision Variables:
- \( x_i \): Number of units of product \( i \) to fulfill, \( x_i \in \mathbb{Z}_+ \) (non-negative integers)

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
- Index set \( I \): Product Name from table_id: file_0_view_0
- Parameter \( r_i \): Revenue from table_id: file_0_view_0, column Revenue
- Parameter \( d_i \): Demand from table_id: file_0_view_0, column Demand
- Parameter \( s_i \): Initial Inventory from table_id: file_0_view_0, column Initial Inventory