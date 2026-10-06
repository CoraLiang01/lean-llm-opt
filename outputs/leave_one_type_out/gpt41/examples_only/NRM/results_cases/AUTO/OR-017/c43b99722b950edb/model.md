ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products i classified under 'ZZ' (from [SKU], table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit for product i (from [Revenue], table_id: file_0_view_0)
- s_i: Initial inventory for product i (from [Initial Inventory], table_id: file_0_view_0)
- d_i: Demand for product i (from [Demand], table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i}, ∀i ∈ I)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
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
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All [SKU] where [SKU] starts with 'ZZ' (table_id: file_0_view_0)
- r_i: [Revenue] (table_id: file_0_view_0)
- s_i: [Initial Inventory] (table_id: file_0_view_0)
- d_i: [Demand] (table_id: file_0_view_0)