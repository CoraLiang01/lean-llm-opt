ABSTRACT OPTIMIZATION MODEL

Index Sets:
- I: Set of products classified under id999.
  (Data source: file_0_view_0, column: id_number, filtered where id_number = 'id999')

Parameters:
- r_i: Revenue per unit for product i ∈ I.
  (Data source: file_0_view_0, column: Revenue)
- d_i: Demand for product i ∈ I during the sales horizon.
  (Data source: file_0_view_0, column: Demand)
- s_i: Initial inventory for product i ∈ I.
  (Data source: file_0_view_0, column: Initial Inventory)

Decision Variables:
- x_i: Number of units of product i ∈ I to fulfill.
  Domain: x_i ∈ ℤ₊ (non-negative integers)

Objective:
- Maximize total revenue:
  max ∑_{i ∈ I} r_i x_i

Constraints:
1. Inventory constraint:
  x_i ≤ s_i  ∀ i ∈ I
2. Demand constraint:
  x_i ≤ d_i  ∀ i ∈ I
3. Non-negativity and integrality:
  x_i ∈ ℤ₊  ∀ i ∈ I

Data Mapping:
- Index set I: file_0_view_0, column id_number, filtered where id_number = 'id999'
- Parameter r_i: file_0_view_0, column Revenue
- Parameter d_i: file_0_view_0, column Demand
- Parameter s_i: file_0_view_0, column Initial Inventory

No literal record values or counts are included. All data sources are mapped by exact table_id and column name.