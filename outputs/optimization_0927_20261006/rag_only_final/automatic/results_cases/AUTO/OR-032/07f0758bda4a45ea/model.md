ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products classified as 'Books'  
  (Source: DifferentStoreSales.csv, table_id: file_0_view_0, column: Product_Name, filtered by prefix 'Books')

Parameters:
- Revenue_i: Revenue per unit for product i ∈ 𝑰  
  (Source: DifferentStoreSales.csv, table_id: file_0_view_0, column: Revenue)
- InitialInventory_i: Initial inventory available for product i ∈ 𝑰  
  (Source: DifferentStoreSales.csv, table_id: file_0_view_0, column: Initial Inventory)
- Demand_i: Deterministic demand for product i ∈ 𝑰  
  (Source: DifferentStoreSales.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill  
  Domain: x_i ≥ 0 and integer

Objective:
- Maximize total revenue from 'Books' products:
  maximize ∑_{i ∈ 𝑰} Revenue_i · x_i

Constraints:
1. Inventory constraint: x_i ≤ InitialInventory_i  ∀ i ∈ 𝑰
2. Demand constraint:  x_i ≤ Demand_i        ∀ i ∈ 𝑰
3. Nonnegativity and integrality: x_i ≥ 0 and integer  ∀ i ∈ 𝑰

Data Mapping:
- Index set 𝑰: DifferentStoreSales.csv, table_id: file_0_view_0, column: Product_Name, filtered by prefix 'Books'
- Revenue_i: DifferentStoreSales.csv, table_id: file_0_view_0, column: Revenue
- InitialInventory_i: DifferentStoreSales.csv, table_id: file_0_view_0, column: Initial Inventory
- Demand_i: DifferentStoreSales.csv, table_id: file_0_view_0, column: Demand