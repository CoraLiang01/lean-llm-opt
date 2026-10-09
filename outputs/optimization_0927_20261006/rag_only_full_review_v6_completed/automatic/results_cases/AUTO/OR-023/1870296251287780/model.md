ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of products classified under ‘ELE-S’, where each product i ∈ 𝑰 is identified by its [Product_Reference].

Parameters:
- Revenue_i: Revenue per unit for product i ∈ 𝑰. (from [Revenue])
- InitialInventory_i: Initial inventory available for product i ∈ 𝑰. (from [Initial Inventory])
- Demand_i: Deterministic demand for product i ∈ 𝑰. (from [Demand])

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill, where x_i ∈ ℤ₊ (non-negative integers).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{InitialInventory_i, Demand_i\} \quad \forall i \in 𝑰
   \]
   (x_i cannot exceed either available inventory or demand for each product.)

Data Mapping:
- Source: SalesStoreoverview.csv, table_id: file_0_view_0
- Index set 𝑰: All records where [Product_Reference] has prefix 'ELE-S' (CSVQA-applied filter: [Product_Reference] prefix 'ELE-S')
- Parameters:
    - Revenue_i: [Revenue] column, file_0_view_0
    - InitialInventory_i: [Initial Inventory] column, file_0_view_0
    - Demand_i: [Demand] column, file_0_view_0

No additional constraints or requirements are imposed beyond those specified in the user query and the returned data.