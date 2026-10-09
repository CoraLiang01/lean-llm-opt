ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of car models classified under 'FDK57' (from BigMartSales.csv, Product Name with prefix 'FDK57').

Parameters:
- Revenue_i: Revenue per unit for car model i ∈ 𝑰 (BigMartSales.csv, Revenue).
- InitialInventory_i: Initial inventory available for car model i ∈ 𝑰 (BigMartSales.csv, Initial Inventory).
- Demand_i: Demand quantity for car model i ∈ 𝑰 (BigMartSales.csv, Demand).

Decision Variables:
- x_i: Number of units of car model i ∈ 𝑰 to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i}).

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
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Table: BigMartSales.csv (table_id: file_0_view_0)
  - Index set 𝑰: All rows where 'Product Name' has prefix 'FDK57'
  - Revenue_i: 'Revenue' column
  - InitialInventory_i: 'Initial Inventory' column
  - Demand_i: 'Demand' column