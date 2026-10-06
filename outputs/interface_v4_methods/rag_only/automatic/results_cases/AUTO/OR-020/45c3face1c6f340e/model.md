Abstract Optimization Model

Index Sets:
- 𝑃: Set of all products, indexed by i. (Product Name from table_id: file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i. (Revenue from file_0_view_0)
- d_i: Demand for product i. (Demand from file_0_view_0)
- s_i: Initial inventory available for product i. (Initial Inventory from file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ 𝑃. (x_i ≥ 0, integer or continuous as appropriate)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in P} r_i \cdot x_i
  \]

Constraints:
1. Demand fulfillment cannot exceed demand:
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
2. Demand fulfillment cannot exceed available inventory:
   \[
   x_i \leq s_i \quad \forall i \in P
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in P
   \]

Data Mapping:
- Index set 𝑃: file_0_view_0, column 'Product Name'
- Parameter r_i: file_0_view_0, column 'Revenue'
- Parameter d_i: file_0_view_0, column 'Demand'
- Parameter s_i: file_0_view_0, column 'Initial Inventory'