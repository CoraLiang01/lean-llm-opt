ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products classified under ‘id999’ (from table_id: file_0_view_0, column: id_number).

Parameters:
- Revenue_i: Revenue per unit for product i ∈ 𝑰 (table_id: file_0_view_0, column: Revenue).
- InitialInventory_i: Initial inventory available for product i ∈ 𝑰 (table_id: file_0_view_0, column: Initial Inventory).
- Demand_i: Demand for product i ∈ 𝑰 during the sales horizon (table_id: file_0_view_0, column: Demand).

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill; x_i ∈ ℤ₊ (non-negative integers).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in 𝑰
   \]
2. Demand constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from table_id: file_0_view_0 in OnlineRetailSalesDataset.csv, using columns:
    - id_number (filtered to 'id999' products) → 𝑰
    - Revenue → Revenue_i
    - Initial Inventory → InitialInventory_i
    - Demand → Demand_i