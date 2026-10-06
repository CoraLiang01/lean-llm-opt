ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all baked goods (indexed by i ∈ 𝑰).

Parameters:
- Revenue_i: Revenue per unit of baked good i. (Source: table_id='file_0_view_0', column='Revenue')
- InitialInventory_i: Initial inventory available for baked good i. (Source: table_id='file_0_view_0', column='Initial Inventory')
- Demand_i: Deterministic demand for baked good i. (Source: table_id='file_0_view_0', column='Demand')

Decision Variables:
- x_i: Quantity of baked good i to fulfill (integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} \text{Revenue}_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment:
   \[
   0 \leq x_i \leq \min\{\text{InitialInventory}_i, \text{Demand}_i\} \quad \forall i \in 𝑰
   \]
   (Or, equivalently, two constraints per i:)
   \[
   x_i \leq \text{InitialInventory}_i \quad \forall i \in 𝑰
   \]
   \[
   x_i \leq \text{Demand}_i \quad \forall i \in 𝑰
   \]
   \[
   x_i \geq 0 \quad \forall i \in 𝑰
   \]
   (x_i can be integer or continuous, as appropriate for the bakery's fulfillment policy.)

Data Mapping:
- Index set 𝑰: All rows in table_id='file_0_view_0', column='Product Name'
- Revenue_i: table_id='file_0_view_0', column='Revenue'
- InitialInventory_i: table_id='file_0_view_0', column='Initial Inventory'
- Demand_i: table_id='file_0_view_0', column='Demand'