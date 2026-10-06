ABSTRACT OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of baked goods, indexed by i (corresponds to all unique values in Frenchbakerydailysales.csv, column Product Name)

Parameters:
- r_i: Revenue per unit of baked good i (Frenchbakerydailysales.csv, column Revenue)
- s_i: Initial inventory available for baked good i (Frenchbakerydailysales.csv, column Initial Inventory)
- d_i: Demand for baked good i (Frenchbakerydailysales.csv, column Demand)

Decision Variables:
- x_i: Quantity of baked good i to fulfill (continuous, x_i ≥ 0, ∀i ∈ 𝑰)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in 𝑰} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in 𝑰
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in 𝑰
   \]
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in 𝑰
   \]

Data Mapping:
- Index set 𝑰: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Product Name
- Parameter r_i: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Revenue
- Parameter s_i: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Initial Inventory
- Parameter d_i: Frenchbakerydailysales.csv, table_id: file_0_view_0, column: Demand