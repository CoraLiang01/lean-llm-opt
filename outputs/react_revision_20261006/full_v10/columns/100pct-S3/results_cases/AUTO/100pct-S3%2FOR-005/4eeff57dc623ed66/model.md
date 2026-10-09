Mathematical Model

Index Sets:
I: set of bread types, from file_1_view_0.item_name

Parameters:
v_i: expected profit per unit of bread type i, from file_1_view_0.item_value
a_i: storage space required per unit of bread type i, from file_1_view_0.resource_requirement
C: total available storage capacity, from file_0_view_0.resource_capacity

Decision Variables:
x_i: number of units of bread type i to order each day, integer and x_i ≥ 0

Objective:
maximize  ∑_{i ∈ I} v_i x_i

Constraint:
∑_{i ∈ I} a_i x_i ≤ C

x_i ∈ ℤ_≥0  for all i ∈ I

Data Mapping

Index Sets:
I: file_1_view_0.item_name

Parameters:
v_i: file_1_view_0.item_value, mapped by item_name
a_i: file_1_view_0.resource_requirement, mapped by item_name
C: file_0_view_0.resource_capacity

Decision Variables:
x_i: number of units of bread type i to order each day, integer, indexed by file_1_view_0.item_name