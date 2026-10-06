Mathematical Optimization Model

Index Sets:
- 𝑰: Set of all “4U” products, indexed by i. (From OnlineSalesinUSA.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit of product i. (file_0_view_0, Revenue)
- d_i: Demand for product i during the sales horizon. (file_0_view_0, Demand)
- s_i: Initial inventory of product i. (file_0_view_0, Initial Inventory)

Decision Variables:
- x_i: Number of units of product i to fulfill, ∀i ∈ 𝑰. (x_i ∈ ℤ₊, i.e., non-negative integers)

Objective:
\[
\max \sum_{i \in 𝑰} r_i \cdot x_i
\]

Constraints:
1. Inventory and demand fulfillment limits:
   \[
   0 \leq x_i \leq \min\{s_i, d_i\} \quad \forall i \in 𝑰
   \]
   (x_i must not exceed either available inventory or realized demand for each product.)

2. Integrality:
   \[
   x_i \in \mathbb{Z}_+ \quad \forall i \in 𝑰
   \]

Data Mapping

- Index set 𝑰: All records in file_0_view_0, column Product Name.
- Parameter r_i: file_0_view_0, column Revenue.
- Parameter d_i: file_0_view_0, column Demand.
- Parameter s_i: file_0_view_0, column Initial Inventory.