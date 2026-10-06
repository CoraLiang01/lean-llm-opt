ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products classified as 'Books'  
  (from DifferentStoreSales.csv, rows where Product_Name has prefix 'Books')

Parameters:
- R_i: Revenue per unit of product i ∈ 𝑰  
  (from DifferentStoreSales.csv, column 'Revenue')
- S_i: Initial inventory of product i ∈ 𝑰  
  (from DifferentStoreSales.csv, column 'Initial Inventory')
- D_i: Demand for product i ∈ 𝑰  
  (from DifferentStoreSales.csv, column 'Demand')

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill  
  Domain: integer, 0 ≤ x_i ≤ min{S_i, D_i}

Objective:
- Maximize total revenue from fulfilled units:
  maximize ∑_{i ∈ 𝑰} R_i · x_i

Constraints:
1. Inventory constraint: x_i ≤ S_i  ∀ i ∈ 𝑰
2. Demand constraint:  x_i ≤ D_i  ∀ i ∈ 𝑰
3. Nonnegativity:    x_i ≥ 0   ∀ i ∈ 𝑰
4. Integrality:      x_i ∈ ℤ   ∀ i ∈ 𝑰

Data Mapping:
- Index set 𝑰: DifferentStoreSales.csv, table_id: file_0_view_0, rows where Product_Name has prefix 'Books'
- R_i: DifferentStoreSales.csv, table_id: file_0_view_0, column 'Revenue'
- S_i: DifferentStoreSales.csv, table_id: file_0_view_0, column 'Initial Inventory'
- D_i: DifferentStoreSales.csv, table_id: file_0_view_0, column 'Demand'