Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑃: Set of all products, where each product p ∈ 𝑃 is identified by 'Product Name' in table_id file_0_view_0.

Parameters (all indexed by p ∈ 𝑃):
- r_p: Revenue per unit of product p ('Revenue', file_0_view_0)
- d_p: Demand for product p ('Demand', file_0_view_0)
- s_p: Initial inventory available for product p ('Initial Inventory', file_0_view_0)

Decision Variables:
- x_p: Quantity of product p to fulfill (continuous, x_p ≥ 0)

Objective:
- Maximize total revenue:
  maximize ∑_{p ∈ 𝑃} r_p · x_p

Constraints:
1. Demand fulfillment cannot exceed demand:
  x_p ≤ d_p  ∀ p ∈ 𝑃
2. Inventory allocation cannot exceed available stock:
  x_p ≤ s_p  ∀ p ∈ 𝑃
3. Non-negativity:
  x_p ≥ 0   ∀ p ∈ 𝑃

Data Mapping:
- Index set 𝑃: All 'Product Name' entries from table_id file_0_view_0
- Parameter r_p: 'Revenue' column, table_id file_0_view_0
- Parameter d_p: 'Demand' column, table_id file_0_view_0
- Parameter s_p: 'Initial Inventory' column, table_id file_0_view_0

All parameters are mapped directly from the specified columns in the current CSV file.