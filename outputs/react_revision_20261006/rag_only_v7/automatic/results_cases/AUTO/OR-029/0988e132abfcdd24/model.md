Mathematical Optimization Model

Index Sets:
- 𝑰: Set of all products classified under ‘FAUX’ (from ZARASales.csv, "Product Name", table_id: file_0_view_0).

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (from "Revenue", table_id: file_0_view_0).
- d_i: Demand quantity for product i ∈ 𝑰 (from "Demand", table_id: file_0_view_0).
- s_i: Initial inventory for product i ∈ 𝑰 (from "Initial Inventory", table_id: file_0_view_0).

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i}).

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Inventory and demand fulfillment bounds for each product:
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in 𝑰
   \]
   (x_i integer)

Data Mapping:
- Index set 𝑰: All products in ZARASales.csv where "Product Name" starts with "FAUX" (table_id: file_0_view_0, column: "Product Name").
- Parameter r_i: "Revenue" column, table_id: file_0_view_0.
- Parameter d_i: "Demand" column, table_id: file_0_view_0.
- Parameter s_i: "Initial Inventory" column, table_id: file_0_view_0.
- Variable x_i: Decision variable for each i ∈ 𝑰.

No additional constraints or data sources are imposed by the query.