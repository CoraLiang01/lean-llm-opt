Mathematical Optimization Model

Index Sets:
- I: Set of all TABLET smartphone models, indexed by i. (from Product Name where prefix is "TABLET")

Parameters:
- r_i: Revenue per unit for model i. (from Revenue, table_id: file_0_view_0)
- d_i: Demand quantity for model i. (from Demand, table_id: file_0_view_0)
- s_i: Initial inventory for model i. (from Initial Inventory, table_id: file_0_view_0)

Decision Variables:
- x_i: Number of units of TABLET model i to fulfill, integer, 0 ≤ x_i ≤ min{d_i, s_i} ∀i∈I

Objective:
- Maximize total revenue:
  max ∑_{i∈I} r_i x_i

Constraints:
1. Demand and inventory fulfillment bounds:
  0 ≤ x_i ≤ min{d_i, s_i}  ∀i∈I

Data Mapping:
- Index set I: All records in table_id file_0_view_0, column Product Name, where Product Name starts with "TABLET"
- Parameter r_i: file_0_view_0, column Revenue, mapped to Product Name
- Parameter d_i: file_0_view_0, column Demand, mapped to Product Name
- Parameter s_i: file_0_view_0, column Initial Inventory, mapped to Product Name
- Variable x_i: defined for each i∈I as above

No additional constraints or data sources are imposed by the query.