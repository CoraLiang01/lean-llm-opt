Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- Let 𝑃 be the set of all baked goods (products), indexed by i ∈ 𝑃.
  (𝑃 is defined by all "Product Name" entries in table_id file_0_view_0.)

Parameters:
- r_i: Revenue per unit of product i.  [Source: file_0_view_0, column "Revenue"]
- d_i: Demand quantity for product i.  [Source: file_0_view_0, column "Demand"]
- s_i: Initial inventory available for product i.  [Source: file_0_view_0, column "Initial Inventory"]

Decision Variables:
- x_i: Quantity of product i to fulfill (sell), for all i ∈ 𝑃.
  Domain: 0 ≤ x_i ≤ min{d_i, s_i}, x_i continuous or integer as appropriate.

Objective:
- Maximize total revenue:
  maximize Z = ∑_{i ∈ 𝑃} r_i · x_i

Constraints:
1. Demand and Inventory Fulfillment:
  For all i ∈ 𝑃:
    0 ≤ x_i ≤ min{d_i, s_i}

Data Mapping:
- Index set 𝑃: All "Product Name" in table_id file_0_view_0
- Parameter r_i: "Revenue" in file_0_view_0
- Parameter d_i: "Demand" in file_0_view_0
- Parameter s_i: "Initial Inventory" in file_0_view_0
- Variable x_i: Quantity fulfilled for each i ∈ 𝑃

No additional constraints or boundary semantics are specified by the query. All parameters are bound directly to the current data columns as described.