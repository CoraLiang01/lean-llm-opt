Abstract Optimization Model

Index Sets:
- 𝑰: Set of ‘Aalop’ products (from Product Name in table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (from Revenue, table_id: file_0_view_0)
- d_i: Demand for product i ∈ 𝑰 (from Demand, table_id: file_0_view_0)
- s_i: Initial inventory of product i ∈ 𝑰 (from Initial Inventory, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, parameters r_i, d_i, s_i, and all constraints are sourced from:
  - table_id: file_0_view_0
    - Product Name → 𝑰
    - Revenue → r_i
    - Demand → d_i
    - Initial Inventory → s_i