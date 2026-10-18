ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products, indexed by i (from "Product Name" in table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i (from "Revenue", table_id: file_0_view_0)
- d_i: Demand for product i (from "Demand", table_id: file_0_view_0)
- s_i: Initial inventory for product i (from "Initial Inventory", table_id: file_0_view_0)

Decision Variables:
- x_i: Quantity of product i to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i})

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint for each product:
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
2. Demand constraint for each product:
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\quad x_i \in \mathbb{Z} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰, parameters r_i, d_i, s_i, and all constraints are sourced from table_id: file_0_view_0 in RetailSalesDataset.csv, using columns:
  - Product Name → 𝑰 (product index)
  - Revenue → r_i
  - Demand → d_i
  - Initial Inventory → s_i

No filters were applied; all records in file_0_view_0 are included.