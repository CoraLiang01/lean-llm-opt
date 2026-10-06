Abstract Optimization Model

Index Sets:
- 𝑰: Set of all products classified under 'Baby' (indexed by i).

Parameters:
- 𝑟ᵢ: Revenue per unit of product i. (from column 'Revenue', table_id: file_0_view_0)
- 𝑑ᵢ: Demand for product i. (from column 'Demand', table_id: file_0_view_0)
- 𝑠ᵢ: Initial inventory of product i. (from column 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑑ᵢ, 𝑠ᵢ})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} rᵢ xᵢ
  \]

Constraints:
1. Demand fulfillment constraint:
   \[
   xᵢ \leq dᵢ \quad \forall i \in 𝑰
   \]
2. Inventory constraint:
   \[
   xᵢ \leq sᵢ \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   xᵢ \geq 0,\ xᵢ \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in table_id: file_0_view_0 (Salesdata.csv) where 'Product Name' starts with 'Baby'
- Parameter 𝑟ᵢ: 'Revenue' column, table_id: file_0_view_0
- Parameter 𝑑ᵢ: 'Demand' column, table_id: file_0_view_0
- Parameter 𝑠ᵢ: 'Initial Inventory' column, table_id: file_0_view_0