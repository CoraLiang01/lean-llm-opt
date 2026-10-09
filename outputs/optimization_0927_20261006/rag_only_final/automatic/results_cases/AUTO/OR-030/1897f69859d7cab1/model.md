ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of car models classified under 'FDK57'  
  (i ∈ 𝑰)

Parameters:
- Revenue_i: Revenue per unit for car model i (from column 'Revenue', table_id: file_0_view_0)
- InitialInventory_i: Initial inventory for car model i (from column 'Initial Inventory', table_id: file_0_view_0)
- Demand_i: Demand quantity for car model i (from column 'Demand', table_id: file_0_view_0)

Decision Variables:
- x_i: Quantity of car model i to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i})

Objective:
- Maximize total revenue:
  maximize ∑_{i ∈ 𝑰} Revenue_i · x_i

Constraints:
1. Inventory constraint:
  x_i ≤ InitialInventory_i  ∀ i ∈ 𝑰
2. Demand fulfillment constraint:
  x_i ≤ Demand_i  ∀ i ∈ 𝑰
3. Non-negativity:
  x_i ≥ 0  ∀ i ∈ 𝑰
4. (If required by context) Integrality:
  x_i ∈ ℤ  ∀ i ∈ 𝑰

Data Mapping:
- Index set 𝑰: All rows in table_id file_0_view_0 where 'Product Name' has prefix 'FDK57'
- Revenue_i: 'Revenue' column, table_id file_0_view_0
- InitialInventory_i: 'Initial Inventory' column, table_id file_0_view_0
- Demand_i: 'Demand' column, table_id file_0_view_0

No literal record values or counts are included. All data references are symbolic and mapped to their exact table_id and column names.