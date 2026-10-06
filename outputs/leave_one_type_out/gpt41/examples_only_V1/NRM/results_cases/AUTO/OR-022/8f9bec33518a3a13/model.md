ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all products classified as ‘27in’, indexed by i ∈ 𝑰.

Parameters:
- Revenue_i: Revenue per unit of product i. (from column ‘Revenue’)
- InitialInventory_i: Initial inventory available for product i. (from column ‘Initial Inventory’)
- Demand_i: Deterministic demand for product i. (from column ‘Demand’)

Decision Variables:
- x_i: Number of units of product i to fulfill, for all i ∈ 𝑰. (x_i ∈ ℤ₊, i.e., non-negative integers)

Objective:
- Maximize total revenue:
  max ∑_{i ∈ 𝑰} Revenue_i × x_i

Constraints:
1. Inventory constraint:
  x_i ≤ InitialInventory_i  ∀ i ∈ 𝑰
2. Demand fulfillment constraint:
  x_i ≤ Demand_i  ∀ i ∈ 𝑰
3. Non-negativity and integrality:
  x_i ∈ ℤ₊  ∀ i ∈ 𝑰

Data Mapping:
- Index set 𝑰: All rows in Salesorders.csv (table_id: file_0_view_0) where ‘Product Name’ has prefix ‘27in’.
- Revenue_i: Salesorders.csv, column ‘Revenue’, table_id: file_0_view_0.
- InitialInventory_i: Salesorders.csv, column ‘Initial Inventory’, table_id: file_0_view_0.
- Demand_i: Salesorders.csv, column ‘Demand’, table_id: file_0_view_0.
- Product identifier: ‘Product Name’, Salesorders.csv, table_id: file_0_view_0.