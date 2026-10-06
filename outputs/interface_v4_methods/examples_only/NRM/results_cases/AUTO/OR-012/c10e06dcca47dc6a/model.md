ABSTRACT MATHEMATICAL OPTIMIZATION MODEL

Index Sets:
- 𝑰: Set of products, indexed by i.

Parameters:
- r_i: Per-unit revenue for product i. (Source: Revenue)
- d_i: Deterministic demand for product i. (Source: Demand)
- s_i: Initial inventory available for product i. (Source: Initial Inventory)

Decision Variables:
- x_i: Number of units of product i to fulfill for customer purchases.
 Domain: Integer, 0 ≤ x_i ≤ min{d_i, s_i}

Objective:
- Maximize total revenue:
  max ∑_{i ∈ 𝑰} r_i x_i

Constraints:
1. Inventory constraint:
  x_i ≤ s_i  ∀ i ∈ 𝑰
2. Demand constraint:
  x_i ≤ d_i  ∀ i ∈ 𝑰
3. Non-negativity and integrality:
  x_i ≥ 0 and integer ∀ i ∈ 𝑰

Data Mapping:
- Table: OnlineSalesDataset.csv (table_id: file_0_view_0)
 • Product index: Product Name
 • Per-unit revenue: Revenue
 • Initial inventory: Initial Inventory
 • Deterministic demand: Demand