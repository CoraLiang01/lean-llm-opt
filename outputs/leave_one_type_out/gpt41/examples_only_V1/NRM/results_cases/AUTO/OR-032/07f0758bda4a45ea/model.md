ABSTRACT Optimization Model

Index Sets:
- 𝑰: Set of products classified as 'Books' (from DifferentStoreSales.csv, Product_Name with prefix 'Books')

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (DifferentStoreSales.csv, Revenue)
- d_i: Demand quantity for product i ∈ 𝑰 (DifferentStoreSales.csv, Demand)
- s_i: Initial inventory for product i ∈ 𝑰 (DifferentStoreSales.csv, Initial Inventory)

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (x_i ≥ 0, integer or continuous as appropriate)

Objective:
- Maximize total revenue from 'Books' products:
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
- Index set 𝑰, parameters r_i, d_i, s_i, and variable x_i are mapped to rows in DifferentStoreSales.csv (table_id: file_0_view_0) where Product_Name has prefix 'Books'.
- r_i: Revenue column (file_0_view_0, Revenue)
- d_i: Demand column (file_0_view_0, Demand)
- s_i: Initial Inventory column (file_0_view_0, Initial Inventory)
- i: Product_Name column (file_0_view_0, Product_Name) with prefix 'Books'