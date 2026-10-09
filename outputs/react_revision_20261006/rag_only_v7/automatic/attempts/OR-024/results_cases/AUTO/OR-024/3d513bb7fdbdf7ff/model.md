Mathematical Optimization Model

Index Sets:
- I: Set of all products with names starting with 'S700_' (from "Product Name" in table_id file_0_view_0).

Parameters:
- r_i: Revenue per unit of product i ∈ I (from "Revenue" in table_id file_0_view_0).
- d_i: Demand quantity for product i ∈ I (from "Demand" in table_id file_0_view_0).
- s_i: Initial inventory for product i ∈ I (from "Initial Inventory" in table_id file_0_view_0).

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill, integer, 0 ≤ x_i ≤ min{d_i, s_i}.

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Demand fulfillment: For all i ∈ I,
   \[
   0 \leq x_i \leq d_i
   \]
2. Inventory limit: For all i ∈ I,
   \[
   x_i \leq s_i
   \]
3. Integrality: For all i ∈ I,
   \[
   x_i \in \mathbb{Z}_{\geq 0}
   \]

Data Mapping:
- Index set I, and parameters r_i, d_i, s_i are defined from table_id file_0_view_0, using columns:
  - "Product Name" (for I)
  - "Revenue" (for r_i)
  - "Demand" (for d_i)
  - "Initial Inventory" (for s_i)