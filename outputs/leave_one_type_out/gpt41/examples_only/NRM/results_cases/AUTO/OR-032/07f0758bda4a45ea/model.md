ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all products classified as 'Books', indexed by i.

Parameters:
- Revenue_i: Revenue per unit of product i. (Source: DifferentStoreSales.csv, column 'Revenue', table_id: file_0_view_0)
- InitialInventory_i: Initial inventory available for product i. (Source: DifferentStoreSales.csv, column 'Initial Inventory', table_id: file_0_view_0)
- Demand_i: Deterministic demand for product i. (Source: DifferentStoreSales.csv, column 'Demand', table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill, for all i ∈ 𝑰. (Domain: integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i})

Objective:
- Maximize total revenue from 'Books' products:
  \[
  \max \sum_{i \in 𝑰} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in 𝑰
   \]
2. Demand constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in DifferentStoreSales.csv (table_id: file_0_view_0) where 'Product_Name' starts with 'Books'.
- Revenue_i: DifferentStoreSales.csv, column 'Revenue', table_id: file_0_view_0
- InitialInventory_i: DifferentStoreSales.csv, column 'Initial Inventory', table_id: file_0_view_0
- Demand_i: DifferentStoreSales.csv, column 'Demand', table_id: file_0_view_0