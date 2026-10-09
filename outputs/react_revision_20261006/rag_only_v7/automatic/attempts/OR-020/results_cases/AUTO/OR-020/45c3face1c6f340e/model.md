Mathematical Optimization Model

Index Sets:
- 𝑃: Set of all products, indexed by i. (From "Product Name" in table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i. (From "Revenue" in table_id: file_0_view_0)
- d_i: Demand for product i. (From "Demand" in table_id: file_0_view_0)
- s_i: Initial inventory available for product i. (From "Initial Inventory" in table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill, integer, 0 ≤ x_i ≤ min{d_i, s_i}, ∀i ∈ 𝑃.

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑃} r_i \cdot x_i
  \]

Constraints:
1. Demand and inventory fulfillment bounds:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\}, \quad \forall i \in 𝑃
   \]

Data Mapping:
- Index set 𝑃: All "Product Name" entries from table_id: file_0_view_0.
- Parameter r_i: "Revenue" column from table_id: file_0_view_0, mapped to product i.
- Parameter d_i: "Demand" column from table_id: file_0_view_0, mapped to product i.
- Parameter s_i: "Initial Inventory" column from table_id: file_0_view_0, mapped to product i.
- Variable x_i: Decision variable for product i, as described above.