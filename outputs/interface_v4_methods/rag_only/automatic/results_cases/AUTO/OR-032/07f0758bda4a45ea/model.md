ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products classified as ‘Books’ (indexed by i).

Parameters:
- r_i: Revenue per unit of product i. (Source: DifferentStoreSales.csv, column ‘Revenue’)
- s_i: Initial inventory of product i. (Source: DifferentStoreSales.csv, column ‘Initial Inventory’)
- d_i: Demand for product i. (Source: DifferentStoreSales.csv, column ‘Demand’)

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ 𝑰. (Domain: x_i ≥ 0, integer or continuous as appropriate)

Objective:
- Maximize total revenue from Books products:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint: x_i ≤ s_i  ∀i ∈ 𝑰
2. Demand constraint:  x_i ≤ d_i  ∀i ∈ 𝑰
3. Nonnegativity:    x_i ≥ 0   ∀i ∈ 𝑰

Data Mapping:
- Index set 𝑰: All rows in DifferentStoreSales.csv where ‘Product_Name’ has prefix ‘Books’ (table_id: file_0_view_0, column: Product_Name)
- r_i: DifferentStoreSales.csv, column ‘Revenue’, table_id: file_0_view_0
- s_i: DifferentStoreSales.csv, column ‘Initial Inventory’, table_id: file_0_view_0
- d_i: DifferentStoreSales.csv, column ‘Demand’, table_id: file_0_view_0