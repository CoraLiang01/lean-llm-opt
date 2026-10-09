ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all products i classified as ‘FAUX’ (from ZARASales.csv, table_id: file_0_view_0, column: Product Name).

Parameters:
- r_i: Revenue per unit of product i (table_id: file_0_view_0, column: Revenue).
- s_i: Initial inventory of product i (table_id: file_0_view_0, column: Initial Inventory).
- d_i: Demand for product i (table_id: file_0_view_0, column: Demand).

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ 𝑰. (Domain: integer, 0 ≤ x_i ≤ min{s_i, d_i})

Objective:
- Maximize total revenue:
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
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in ZARASales.csv (table_id: file_0_view_0) where Product Name contains or starts with 'FAUX'.
- r_i: Revenue (table_id: file_0_view_0, column: Revenue)
- s_i: Initial Inventory (table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand (table_id: file_0_view_0, column: Demand)