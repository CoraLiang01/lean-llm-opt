ABSTRACT OPTIMIZATION MODEL

Index Sets:
- Let 𝑃 be the set of all pizza types, indexed by i.
  (Source: PizzaSalesDataset.csv, table_id: file_0_view_0, column: Product Name)

Parameters:
- r_i: Revenue per unit of pizza type i
  (Source: PizzaSalesDataset.csv, table_id: file_0_view_0, column: Revenue)
- s_i: Initial inventory available for pizza type i
  (Source: PizzaSalesDataset.csv, table_id: file_0_view_0, column: Initial Inventory)
- d_i: Demand for pizza type i during the sales horizon
  (Source: PizzaSalesDataset.csv, table_id: file_0_view_0, column: Demand)

Decision Variables:
- x_i: Number of units of pizza type i to fulfill (integer, x_i ≥ 0)

Objective:
- Maximize total revenue:
  maximize Z = ∑_{i ∈ P} r_i x_i

Constraints:
1. Inventory constraint:
  x_i ≤ s_i  ∀ i ∈ P
2. Demand constraint:
  x_i ≤ d_i  ∀ i ∈ P
3. Non-negativity and integrality:
  x_i ∈ {0, 1, 2, ...}  ∀ i ∈ P

Data Mapping:
- Index set P (pizza types): PizzaSalesDataset.csv, table_id: file_0_view_0, column: Product Name
- Parameter r_i (revenue): PizzaSalesDataset.csv, table_id: file_0_view_0, column: Revenue
- Parameter s_i (initial inventory): PizzaSalesDataset.csv, table_id: file_0_view_0, column: Initial Inventory
- Parameter d_i (demand): PizzaSalesDataset.csv, table_id: file_0_view_0, column: Demand

This model maximizes total revenue by optimally fulfilling pizza orders, subject to both inventory and demand limits for each pizza type.