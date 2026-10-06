Mathematical Optimization Model

Index Sets:
- 𝑃: Set of products, indexed by i. (Product Name from file_0_view_0)

Parameters:
- r_i: Revenue per unit of product i. (Revenue from file_0_view_0)
- d_i: Demand quantity for product i. (Demand from file_0_view_0)
- s_i: Initial inventory available for product i. (Initial Inventory from file_0_view_0)

Decision Variables:
- x_i: Number of units of product i to fulfill, integer, 0 ≤ x_i ≤ min{d_i, s_i} ∀i ∈ 𝑃

Objective:
- Maximize total revenue:
\[
\max \sum_{i \in 𝑃} r_i \cdot x_i
\]

Constraints:
1. Demand fulfillment cannot exceed demand or inventory:
\[
0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in 𝑃
\]

Data Mapping:
- Index set 𝑃: file_0_view_0, column Product Name
- Parameter r_i: file_0_view_0, column Revenue
- Parameter d_i: file_0_view_0, column Demand
- Parameter s_i: file_0_view_0, column Initial Inventory

Notes:
- All variables x_i are integer and bounded as above.
- All data is mapped directly from the specified columns in file_0_view_0.