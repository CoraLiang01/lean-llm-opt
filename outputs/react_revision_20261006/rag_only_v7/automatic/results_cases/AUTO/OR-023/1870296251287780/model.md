Mathematical Optimization Model (Abstract Formulation)

Index Sets:
- 𝑰: Set of all products classified under ‘ELE-S’ (from column Product_Reference in table_id file_0_view_0).

Parameters:
- r_i: Revenue per unit of product i ∈ 𝑰 (from column Revenue, table_id file_0_view_0).
- d_i: Demand quantity for product i ∈ 𝑰 (from column Demand, table_id file_0_view_0).
- s_i: Initial inventory for product i ∈ 𝑰 (from column Initial Inventory, table_id file_0_view_0).

Decision Variables:
- x_i: Number of units of product i ∈ 𝑰 to fulfill (integer, 0 ≤ x_i ≤ min{d_i, s_i}).

Objective:
- Maximize total revenue:
  maximize ∑_{i ∈ 𝑰} r_i · x_i

Constraints:
1. Inventory and demand fulfillment bounds:
  0 ≤ x_i ≤ min{d_i, s_i}  ∀ i ∈ 𝑰

Data Mapping:
- Index set 𝑰: All Product_Reference values in file_0_view_0 where Product_Reference starts with 'ELE-S'
- Parameter r_i: Revenue column in file_0_view_0, mapped by Product_Reference
- Parameter d_i: Demand column in file_0_view_0, mapped by Product_Reference
- Parameter s_i: Initial Inventory column in file_0_view_0, mapped by Product_Reference
- Variable x_i: Decision variable for each i ∈ 𝑰

No additional constraints or data sources are imposed by the query. All bounds are unconditional.