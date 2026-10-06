ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of car models classified under ‘FDK57’. (i ∈ I)

Parameters:
- Revenue_i: Revenue per unit for car model i. [table_id: file_0_view_0, column: Revenue]
- InitialInventory_i: Initial inventory available for car model i. [table_id: file_0_view_0, column: Initial Inventory]
- Demand_i: Demand quantity for car model i. [table_id: file_0_view_0, column: Demand]

Decision Variables:
- x_i: Quantity of car model i to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i}), ∀i ∈ I

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I and all parameters (Revenue_i, InitialInventory_i, Demand_i) are sourced from table_id: file_0_view_0 (BigMartSales.csv), with the following columns:
    - Car model: 'Product Name' (filtered for prefix 'FDK57')
    - Revenue: 'Revenue'
    - Initial Inventory: 'Initial Inventory'
    - Demand: 'Demand'