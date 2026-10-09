ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products classified as 'Books' (from DifferentStoreSales.csv where Product_Name has prefix 'Books')

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (from Revenue, table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory of product i ∈ 𝑰 (from Initial Inventory, table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for product i ∈ 𝑰 (from Demand, table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (x_i ≥ 0, integer or continuous as appropriate)

Objective:
- Maximize total revenue from fulfilled units:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in DifferentStoreSales.csv (table_id: file_0_view_0) where Product_Name has prefix 'Books'
- r_i: Revenue (column: Revenue, table_id: file_0_view_0)
- s_i: Initial Inventory (column: Initial Inventory, table_id: file_0_view_0)
- d_i: Demand (column: Demand, table_id: file_0_view_0)