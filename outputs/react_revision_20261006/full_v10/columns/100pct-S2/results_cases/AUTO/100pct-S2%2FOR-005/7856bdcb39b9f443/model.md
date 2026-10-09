Mathematical Model

Index Sets:
Let I be the set of bread types, with each element i identified by file_1_view_0:item_name.

Parameters:
Let v_i = file_1_view_0:item_value, the expected profit per unit of bread type i.
Let a_i = file_1_view_0:resource_requirement, the storage space required per unit of bread type i.
Let C = file_0_view_0:resource_capacity, the total available storage capacity.

Decision Variables:
For each i in I, let x_i ∈ ℤ≥0 be the number of units of bread type i to order each day.

Objective:
Maximize total expected profit:
max ∑_{i∈I} v_i x_i

Constraint:
Total storage used cannot exceed available capacity:
∑_{i∈I} a_i x_i ≤ C

Variable Domains:
x_i ∈ ℤ≥0 for all i ∈ I

Data Mapping

Index Sets:
I = {file_1_view_0:item_name}

Parameters:
v_i = file_1_view_0:item_value, for i ∈ I
a_i = file_1_view_0:resource_requirement, for i ∈ I
C = file_0_view_0:resource_capacity

Decision Variables:
x_i: number of units to order of bread type i, integer, for i ∈ I

Objective:
max ∑_{i∈I} v_i x_i

Constraint:
∑_{i∈I} a_i x_i ≤ C

Variable Domains:
x_i ∈ ℤ≥0 for all i ∈ I