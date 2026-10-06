Abstract Optimization Model

Index Sets:
- 𝑰: Set of car models classified under 'FDK57' (from Product Name in file_0_view_0).

Parameters:
- r_i: Revenue per unit for car model i ∈ 𝑰 (Revenue, file_0_view_0).
- s_i: Initial inventory for car model i ∈ 𝑰 (Initial Inventory, file_0_view_0).
- d_i: Demand for car model i ∈ 𝑰 (Demand, file_0_view_0).

Decision Variables:
- x_i: Quantity of car model i ∈ 𝑰 to fulfill (integer, 0 ≤ x_i ≤ min{s_i, d_i}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i x_i
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
- Index set 𝑰, and parameters r_i, s_i, d_i are sourced from table_id: file_0_view_0, columns: 'Product Name' (for 𝑰), 'Revenue' (for r_i), 'Initial Inventory' (for s_i), and 'Demand' (for d_i).