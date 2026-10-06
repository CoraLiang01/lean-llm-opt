ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all products classified as ‘FAUX’ (from ZARASales.csv, table_id: file_0_view_0, column: Product Name).

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (table_id: file_0_view_0, column: Revenue).
- s_i: Initial inventory of product i ∈ 𝑰 (table_id: file_0_view_0, column: Initial Inventory).
- d_i: Demand for product i ∈ 𝑰 (table_id: file_0_view_0, column: Demand).

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (integer, x_i ≥ 0).

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
   x_i \in \mathbb{Z}_+, \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, and parameters r_i, s_i, d_i are sourced from ZARASales.csv (table_id: file_0_view_0), using:
    - Product Name (for 𝑰)
    - Revenue (for r_i)
    - Initial Inventory (for s_i)
    - Demand (for d_i)
  Filter: Only rows where Product Name contains or starts with ‘FAUX’.