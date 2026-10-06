Abstract Mathematical Optimization Model

Index Sets:
- 𝑰: Set of products classified under ‘id999’. (i ∈ 𝑰)

Parameters:
- Revenue_i: Revenue per unit for product i. [Source: OnlineRetailSalesDataset.csv, column ‘Revenue’, table_id: file_0_view_0]
- InitialInventory_i: Initial inventory available for product i. [Source: OnlineRetailSalesDataset.csv, column ‘Initial Inventory’, table_id: file_0_view_0]
- Demand_i: Demand for product i during the sales horizon. [Source: OnlineRetailSalesDataset.csv, column ‘Demand’, table_id: file_0_view_0]

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, x_i ≥ 0).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} Revenue_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq InitialInventory_i \quad \forall i \in 𝑰
   \]
2. Demand constraint:
   \[
   x_i \leq Demand_i \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: All rows in OnlineRetailSalesDataset.csv where ‘id_number’ = ‘id999’ (table_id: file_0_view_0, column: ‘id_number’).
- Revenue_i: OnlineRetailSalesDataset.csv, column ‘Revenue’, table_id: file_0_view_0.
- InitialInventory_i: OnlineRetailSalesDataset.csv, column ‘Initial Inventory’, table_id: file_0_view_0.
- Demand_i: OnlineRetailSalesDataset.csv, column ‘Demand’, table_id: file_0_view_0.