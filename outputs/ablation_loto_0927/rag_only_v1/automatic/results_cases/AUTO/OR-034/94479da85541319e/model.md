ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all baked goods (indexed by i ∈ 𝑰).

Parameters:
- Revenue_i: Revenue per unit of baked good i (from column 'Revenue', table_id: file_0_view_0).
- InitialInventory_i: Initial inventory available for baked good i (from column 'Initial Inventory', table_id: file_0_view_0).
- Demand_i: Deterministic demand for baked good i (from column 'Demand', table_id: file_0_view_0).

Decision Variables:
- x_i: Quantity of baked good i to fulfill (continuous, x_i ≥ 0, ∀i ∈ 𝑰).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} \text{Revenue}_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq \text{InitialInventory}_i \quad \forall i \in 𝑰
   \]
2. Demand constraint:
   \[
   x_i \leq \text{Demand}_i \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All unique values in 'Product Name' (table_id: file_0_view_0).
- Revenue_i: 'Revenue' column (table_id: file_0_view_0).
- InitialInventory_i: 'Initial Inventory' column (table_id: file_0_view_0).
- Demand_i: 'Demand' column (table_id: file_0_view_0).