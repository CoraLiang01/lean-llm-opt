Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of all products, indexed by i. (From "Product Name" in table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i. (From "Revenue", table_id: file_0_view_0)
- d_i: Demand for product i (deterministic, known). (From "Demand", table_id: file_0_view_0)
- s_i: Initial inventory available for product i. (From "Initial Inventory", table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue:
  \[
  \max_{x} \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Demand fulfillment cannot exceed demand or inventory:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in 𝑰
   \]
   (If x_i must be integer, add: x_i ∈ ℤ₊)

Data Mapping:
- Index set 𝑰: All "Product Name" entries in table_id: file_0_view_0
- Parameter r_i: "Revenue" column, table_id: file_0_view_0
- Parameter d_i: "Demand" column, table_id: file_0_view_0
- Parameter s_i: "Initial Inventory" column, table_id: file_0_view_0
- Decision variable x_i: Number of units fulfilled for product i

All parameters are directly mapped from the specified columns in the current CSV file. The model maximizes total revenue subject to inventory and demand limits for each product.