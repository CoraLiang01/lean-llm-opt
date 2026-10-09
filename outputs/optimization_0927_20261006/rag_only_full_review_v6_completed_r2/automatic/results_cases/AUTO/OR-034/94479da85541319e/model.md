ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑰 be the set of all baked goods, indexed by i ∈ 𝑰.

Parameters:
- r_i: Revenue per unit of baked good i (from column 'Revenue', table_id: file_0_view_0)
- d_i: Demand for baked good i (from column 'Demand', table_id: file_0_view_0)
- s_i: Initial inventory for baked good i (from column 'Initial Inventory', table_id: file_0_view_0)

Decision Variables:
- x_i ≥ 0: Quantity of baked good i to fulfill (continuous, for all i ∈ 𝑰)

Objective:
Maximize total revenue:
\[
\max \sum_{i \in 𝑰} r_i \cdot x_i
\]

Constraints:
1. Demand fulfillment constraint:
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
2. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
3. Nonnegativity:
   \[
   x_i \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰 and all parameters (r_i, d_i, s_i) are sourced from table_id: file_0_view_0, columns: 'Product Name', 'Revenue', 'Demand', 'Initial Inventory' in Frenchbakerydailysales.csv. No filters were applied; all records are included.