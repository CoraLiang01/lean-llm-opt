ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of all products classified under ‘Baby’ (from table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit of product i (from table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory of product i (from table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for product i (from table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of product i to fulfill (continuous or integer, as appropriate; domain: 0 ≤ x_i ≤ min{s_i, d_i}, ∀i ∈ I)

Objective:
- Maximize total revenue:
  \[
  \max \sum_{i \in I} r_i x_i
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
3. Non-negativity:
   \[
   x_i \geq 0 \quad \forall i \in I
   \]

Data Mapping:
- Index set I, and all parameters (r_i, s_i, d_i) are sourced from table_id: file_0_view_0 in EuropeSalesRecords.csv, with columns:
    - Product Name (for I)
    - Revenue (for r_i)
    - Initial Inventory (for s_i)
    - Demand (for d_i)