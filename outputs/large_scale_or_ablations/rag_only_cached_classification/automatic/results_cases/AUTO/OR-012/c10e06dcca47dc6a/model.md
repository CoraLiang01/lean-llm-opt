ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑃: Set of products, indexed by i.

Parameters:
- Revenue_i: Revenue per unit for product i. (Source: OnlineSalesDataset.csv, column 'Revenue')
- InitialInventory_i: Initial inventory available for product i. (Source: OnlineSalesDataset.csv, column 'Initial Inventory')
- Demand_i: Deterministic demand for product i over the sales horizon. (Source: OnlineSalesDataset.csv, column 'Demand')

Decision Variables:
- x_i: Number of units of product i to fulfill for customer purchases.
    Domain: Integer, 0 ≤ x_i ≤ min{InitialInventory_i, Demand_i} for all i ∈ 𝑃.

Objective:
- Maximize total revenue:
    maximize ∑_{i ∈ 𝑃} Revenue_i × x_i

Constraints:
1. Inventory constraint:
  x_i ≤ InitialInventory_i  ∀ i ∈ 𝑃
2. Demand fulfillment constraint:
  x_i ≤ Demand_i  ∀ i ∈ 𝑃
3. Non-negativity and integrality:
  x_i ∈ {0, 1, 2, ..., min{InitialInventory_i, Demand_i}}  ∀ i ∈ 𝑃

Data Mapping:
- Index set 𝑃: All unique values in 'Product Name' from OnlineSalesDataset.csv (table_id: file_0_view_0)
- Revenue_i: 'Revenue' column from OnlineSalesDataset.csv (table_id: file_0_view_0)
- InitialInventory_i: 'Initial Inventory' column from OnlineSalesDataset.csv (table_id: file_0_view_0)
- Demand_i: 'Demand' column from OnlineSalesDataset.csv (table_id: file_0_view_0)
- Decision variable x_i: defined for each i ∈ 𝑃

No additional relationships or tables are required. All data is sourced from OnlineSalesDataset.csv (table_id: file_0_view_0).