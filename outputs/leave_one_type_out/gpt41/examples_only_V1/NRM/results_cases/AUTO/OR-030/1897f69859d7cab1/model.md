ABSTRACT Optimization Model

Index Sets:
- 𝑰: Set of car models classified under 'FDK57' (from Product Name column where prefix is 'FDK57').

Parameters:
- Revenue_i: Revenue per unit for car model i ∈ 𝑰 (from Revenue column).
- InitialInventory_i: Initial inventory for car model i ∈ 𝑰 (from Initial Inventory column).
- Demand_i: Demand quantity for car model i ∈ 𝑰 (from Demand column).

Decision Variables:
- x_i: Quantity of car model i ∈ 𝑰 to fulfill (continuous, with 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each car model:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in 𝑰
   \]
2. Demand fulfillment constraint for each car model:
   \[
   x_i \leq Demand_i \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Table: file_0_view_0 (from BigMartSales.csv)
  - Index set 𝑰: All rows where Product Name has prefix 'FDK57'
  - Revenue_i: 'Revenue' column
  - InitialInventory_i: 'Initial Inventory' column
  - Demand_i: 'Demand' column