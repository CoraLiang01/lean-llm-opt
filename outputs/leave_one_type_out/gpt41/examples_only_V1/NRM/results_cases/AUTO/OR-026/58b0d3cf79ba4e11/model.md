ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all products classified under ‘Fashion’. (From SupermarketSales.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- 𝑟ᵢ: Revenue per unit of product i ∈ 𝑰. (SupermarketSales.csv, table_id: file_0_view_0, column: Revenue)
- 𝑠ᵢ: Initial inventory of product i ∈ 𝑰. (SupermarketSales.csv, table_id: file_0_view_0, column: Initial Inventory)
- 𝑑ᵢ: Demand quantity for product i ∈ 𝑰. (SupermarketSales.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- 𝑥ᵢ: Number of units of product i ∈ 𝑰 to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑠ᵢ, 𝑑ᵢ})

Objective:
- Maximize total revenue from fulfilled Fashion product demand:
  \[
  \max \sum_{i \in 𝑰} r_i x_i
  \]

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in 𝑰
   \]
   (Each product’s fulfilled quantity cannot exceed available inventory or demand.)

2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, and all parameters (𝑟ᵢ, 𝑠ᵢ, 𝑑ᵢ) are sourced from SupermarketSales.csv, table_id: file_0_view_0, columns: Product Name, Revenue, Initial Inventory, Demand, filtered for Fashion products.