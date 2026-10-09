ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all Fashion products, indexed by i.

Parameters:
- 𝑟ᵢ: Revenue per unit of product i. (Source: SupermarketSales.csv, column 'Revenue')
- 𝑠ᵢ: Initial inventory of product i. (Source: SupermarketSales.csv, column 'Initial Inventory')
- 𝑑ᵢ: Demand for product i. (Source: SupermarketSales.csv, column 'Demand')

Decision Variables:
- 𝑥ᵢ: Number of units of Fashion product i to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑠ᵢ, 𝑑ᵢ})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} rᵢ xᵢ
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   xᵢ \leq sᵢ \quad \forall i \in 𝑰
   \]
2. Demand fulfillment constraint for each product:
   \[
   xᵢ \leq dᵢ \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   xᵢ \geq 0,\ xᵢ \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, and parameters 𝑟ᵢ, 𝑠ᵢ, 𝑑ᵢ are sourced from table_id: file_0_view_0 (SupermarketSales.csv), columns: 'Product Name' (for index), 'Revenue', 'Initial Inventory', 'Demand', filtered where 'Product Name' has prefix 'Fashion'.