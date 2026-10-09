ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let I = {i} be the set of products classified under ‘id999’. (From the data, I contains the single product with id_number = 'id999'.)

Parameters:
- Revenue_i: Revenue per unit for product i ∈ I. (From column 'Revenue', table_id: file_0_view_0)
- InitialInventory_i: Initial inventory available for product i ∈ I. (From column 'Initial Inventory', table_id: file_0_view_0)
- Demand_i: Demand during the sales horizon for product i ∈ I. (From column 'Demand', table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill, where x_i ∈ ℤ₊ (non-negative integer).

Objective:
- Maximize total revenue from fulfilled quantities:
  \[
  \max \sum_{i \in I} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint: Fulfilled quantity cannot exceed initial inventory.
   \[
   x_i \leq InitialInventory_i \quad \forall i \in I
   \]
2. Demand constraint: Fulfilled quantity cannot exceed demand.
   \[
   x_i \leq Demand_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All records in OnlineRetailSalesDataset.csv (table_id: file_0_view_0) where id_number = 'id999'.
- Revenue_i: 'Revenue' column, table_id: file_0_view_0, for i ∈ I.
- InitialInventory_i: 'Initial Inventory' column, table_id: file_0_view_0, for i ∈ I.
- Demand_i: 'Demand' column, table_id: file_0_view_0, for i ∈ I.

Note: The model is written abstractly and supports any number of products matching the filter id_number = 'id999', though the current data contains only one such product. All constraints and the objective are defined symbolically.