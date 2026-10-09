Abstract Optimization Model for Dairy Product Order Fulfillment

Index Sets:
- 𝑰: Set of all dairy products (indexed by i)

Parameters:
- D_i: Demand for product i (from column Demand, table_id: file_0_view_0)
- S_i: Initial inventory for product i (from column Initial Inventory, table_id: file_0_view_0)
- p_i: Revenue per unit for product i (from column Revenue, table_id: file_0_view_0)

Decision Variables:
- x_i ≥ 0: Number of units of product i to fulfill (continuous, for all i ∈ 𝑰)

Objective:
Maximize total revenue:
\[
\max \sum_{i \in 𝑰} p_i \cdot x_i
\]

Constraints:
1. Inventory and demand fulfillment limits:
   \[
   0 \leq x_i \leq \min\{D_i, S_i\} \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰 and all parameters (D_i, S_i, p_i) are sourced from table_id: file_0_view_0, columns: Full_Product_Name (for i), Demand (for D_i), Initial Inventory (for S_i), Revenue (for p_i). All records in this table are used; no filters are applied.

This model maximizes total revenue by optimally fulfilling orders for each dairy product, subject to both demand and initial inventory constraints.