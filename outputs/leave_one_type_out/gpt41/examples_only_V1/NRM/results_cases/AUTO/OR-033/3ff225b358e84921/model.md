ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of all products classified under ‘Baby’ (from EuropeSalesRecords.csv, filtered by Product Name prefix "Baby")

Parameters:
- Revenue_i: Revenue per unit of product i ∈ 𝑰 (EuropeSalesRecords.csv, column: Revenue)
- InitialInventory_i: Initial inventory available for product i ∈ 𝑰 (EuropeSalesRecords.csv, column: Initial Inventory)
- Demand_i: Deterministic demand for product i ∈ 𝑰 (EuropeSalesRecords.csv, column: Demand)

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i})

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
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from EuropeSalesRecords.csv (table_id: file_0_view_0), with rows filtered where Product Name has prefix "Baby". Columns used: 
  - Product Name (for index set 𝑰)
  - Revenue (for Revenue_i)
  - Initial Inventory (for InitialInventory_i)
  - Demand (for Demand_i)