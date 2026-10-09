ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let I be the set of car models classified under ‘FDK57’, indexed by i ∈ I.

Parameters:
- Revenue_i: Revenue per unit for car model i. (from column 'Revenue', table_id: file_0_view_0)
- Inventory_i: Initial inventory available for car model i. (from column 'Initial Inventory', table_id: file_0_view_0)
- Demand_i: Deterministic demand for car model i. (from column 'Demand', table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of car model i to fulfill (integer, 0 ≤ x_i ≤ min{Inventory_i, Demand_i}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each model:
   \[
   x_i \leq Inventory_i \quad \forall i \in I
   \]
2. Demand fulfillment constraint for each model:
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0 \text{ and integer} \quad \forall i \in I
   \]

Data Mapping:
- I: All records in table_id: file_0_view_0 where 'Product Name' = 'FDK57'
- Revenue_i: 'Revenue' column, table_id: file_0_view_0
- Inventory_i: 'Initial Inventory' column, table_id: file_0_view_0
- Demand_i: 'Demand' column, table_id: file_0_view_0

This model maximizes total revenue from fulfilling deterministic demand for each FDK57 car model, subject to inventory and demand limits.