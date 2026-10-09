ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all products i classified as ‘FAUX’ (from ZARASales.csv, Product Name contains "FAUX").

Parameters:
- 𝑟ᵢ: Revenue per unit of product i (ZARASales.csv, Revenue, table_id: file_0_view_0, column: Revenue)
- 𝑠ᵢ: Initial inventory of product i (ZARASales.csv, Initial Inventory, table_id: file_0_view_0, column: Initial Inventory)
- 𝑑ᵢ: Demand for product i (ZARASales.csv, Demand, table_id: file_0_view_0, column: Demand)

Decision Variables:
- 𝑥ᵢ: Number of units of product i to fulfill (integer, 0 ≤ 𝑥ᵢ ≤ min{𝑠ᵢ, 𝑑ᵢ}, ∀i ∈ 𝑰)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} rᵢ xᵢ
  \]

Constraints:
1. Inventory constraint:
   \[
   xᵢ \leq sᵢ \quad \forall i \in 𝑰
   \]
2. Demand constraint:
   \[
   xᵢ \leq dᵢ \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   xᵢ \geq 0,\ xᵢ \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in ZARASales.csv (table_id: file_0_view_0) where Product Name contains "FAUX"
- Parameter 𝑟ᵢ: ZARASales.csv, column Revenue, table_id: file_0_view_0
- Parameter 𝑠ᵢ: ZARASales.csv, column Initial Inventory, table_id: file_0_view_0
- Parameter 𝑑ᵢ: ZARASales.csv, column Demand, table_id: file_0_view_0