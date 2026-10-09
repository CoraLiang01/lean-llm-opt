ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of products classified as 'ELE-S' (Product_Reference in file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i ∈ I (Revenue, table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory of product i ∈ I (Initial Inventory, table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for product i ∈ I (Demand, table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in table_id file_0_view_0 where Product_Reference starts with 'ELE-S'
- r_i: Revenue (file_0_view_0, column: Revenue)
- s_i: Initial Inventory (file_0_view_0, column: Initial Inventory)
- d_i: Demand (file_0_view_0, column: Demand)