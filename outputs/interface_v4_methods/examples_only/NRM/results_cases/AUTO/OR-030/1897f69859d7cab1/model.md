ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of car models classified under 'FDK57' (from Product Name column where prefix = 'FDK57', table_id: file_0_view_0)

Parameters:
- Revenue_i: Revenue per unit for car model i ∈ I (from Revenue, table_id: file_0_view_0)
- InitialInventory_i: Initial inventory for car model i ∈ I (from Initial Inventory, table_id: file_0_view_0)
- Demand_i: Demand quantity for car model i ∈ I (from Demand, table_id: file_0_view_0)

Decision Variables:
- x_i: Quantity of car model i ∈ I to fulfill (continuous, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each car model:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in I
   \]
2. Demand fulfillment constraint for each car model:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Index set I, and parameters Revenue_i, InitialInventory_i, Demand_i are sourced from table_id: file_0_view_0, columns: 'Product Name', 'Revenue', 'Initial Inventory', 'Demand' (filtered where 'Product Name' has prefix 'FDK57').