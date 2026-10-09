Mathematical Optimization Model

Index Sets:
- 𝑃: Set of all products, indexed by i. (From "Product Name" in table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i. (From "Revenue" in table_id: file_0_view_0)
- d_i: Deterministic demand for product i. (From "Demand" in table_id: file_0_view_0)
- s_i: Initial inventory available for product i. (From "Initial Inventory" in table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill for customer purchases.
 Domain: Integer, 0 ≤ x_i ≤ min{d_i, s_i}, ∀i ∈ 𝑃

Objective:
Maximize total revenue:
 max ∑_{i ∈ 𝑃} r_i x_i

Constraints:
1. Demand fulfillment and inventory limits:
 0 ≤ x_i ≤ min{d_i, s_i}, ∀i ∈ 𝑃

Data Mapping:
- Index set 𝑃: All "Product Name" entries in table_id: file_0_view_0
- Parameter r_i: "Revenue" column in table_id: file_0_view_0
- Parameter d_i: "Demand" column in table_id: file_0_view_0
- Parameter s_i: "Initial Inventory" column in table_id: file_0_view_0
- Decision variable x_i: Number of units to fulfill for each i ∈ 𝑃

All parameters are directly mapped from the specified columns in the current CSV file. The model maximizes total revenue subject to inventory and demand constraints for each product.