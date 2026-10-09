Mathematical Model

Index Sets:
I: set of bread types, with each element i identified by file_1_view_0:item_name

Parameters:
v_i: expected profit per unit of bread type i (file_1_view_0:item_value)
a_i: storage space required per unit of bread type i (file_1_view_0:resource_requirement)
C: total available storage capacity (file_0_view_0:resource_capacity)

Decision Variables:
x_i: number of units of bread type i to order each day (integer, x_i ≥ 0)

Objective:
maximize  ∑_{i ∈ I} v_i x_i

Constraint:
∑_{i ∈ I} a_i x_i ≤ C

x_i ∈ ℤ_≥0 for all i ∈ I

Data Mapping

Index Sets:
I = all file_1_view_0:item_name

Parameters:
v_i = file_1_view_0:item_value for bread type i
a_i = file_1_view_0:resource_requirement for bread type i
C = file_0_view_0:resource_capacity

Decision Variables:
x_i: integer, nonnegative, for each i ∈ I

Objective:
maximize total expected profit: sum over i of v_i x_i

Constraint:
total storage used: sum over i of a_i x_i ≤ C

Nonnegativity and integrality:
x_i ∈ ℤ_≥0 for all i ∈ I