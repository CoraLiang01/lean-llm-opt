ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all products i classified under 'Fashion' (from SupermarketSales.csv where Product Name starts with 'Fashion').

Parameters:
- Revenue_i: Revenue per unit of product i. [SupermarketSales.csv, column: Revenue, table_id: file_0_view_0]
- InitialInventory_i: Initial inventory available for product i. [SupermarketSales.csv, column: Initial Inventory, table_id: file_0_view_0]
- Demand_i: Deterministic demand for product i. [SupermarketSales.csv, column: Demand, table_id: file_0_view_0]

Decision Variables:
- x_i: Number of units of product i to fulfill (continuous, x_i ≥ 0).

Objective:
- Maximize total revenue from Fashion products:
  \[
  \max \sum_{i \in 𝑰} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in 𝑰
   \]
2. Demand fulfillment constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from SupermarketSales.csv, table_id: file_0_view_0, using rows where [Product Name] has prefix 'Fashion', and columns: [Product Name], [Revenue], [Initial Inventory], [Demand].