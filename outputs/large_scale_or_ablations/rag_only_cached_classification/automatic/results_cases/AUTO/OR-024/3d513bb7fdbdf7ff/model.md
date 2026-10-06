ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products i such that Product Name starts with 'S700_' (from table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit of product i (from table_id: file_0_view_0, column: Revenue)
- d_i: Demand for product i (from table_id: file_0_view_0, column: Demand)
- s_i: Initial inventory of product i (from table_id: file_0_view_0, column: Initial Inventory)

Decision Variables:
- x_i: Number of units of product i to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i}), ∀ i ∈ I

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i \cdot x_i
  \]

Constraints:
1. Inventory constraint:
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\ x_i \in \mathbb{Z} \quad \forall i \in I
   \]

Data Mapping:
- Index set I: All rows in file_0_view_0 where Product Name starts with 'S700_'
- r_i: file_0_view_0, column: Revenue
- d_i: file_0_view_0, column: Demand
- s_i: file_0_view_0, column: Initial Inventory

No data values or record counts are included; all sources are referenced by exact table_id and column name.