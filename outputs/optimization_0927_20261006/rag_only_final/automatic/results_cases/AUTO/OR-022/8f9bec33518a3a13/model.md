ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products i classified under ‘27in’ (from Salesorders.csv, Product Name with prefix '27in')

Parameters:
- Revenue_i: Revenue per unit for product i ∈ I (Salesorders.csv, Revenue)
- InitialInventory_i: Initial inventory available for product i ∈ I (Salesorders.csv, Initial Inventory)
- Demand_i: Deterministic demand for product i ∈ I (Salesorders.csv, Demand)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in I
   \]
2. Demand constraint for each product:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I, and parameters Revenue_i, InitialInventory_i, Demand_i are sourced from Salesorders.csv (table_id: file_0_view_0), using:
  - Product Name (prefix '27in') → I
  - Revenue → Revenue_i
  - Initial Inventory → InitialInventory_i
  - Demand → Demand_i