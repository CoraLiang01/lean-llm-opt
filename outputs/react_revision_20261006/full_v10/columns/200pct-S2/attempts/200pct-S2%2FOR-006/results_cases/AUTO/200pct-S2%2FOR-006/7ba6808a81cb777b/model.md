Mathematical Model

Index Sets:
I: set of vehicle types, as indexed by ProductName in file_1_view_0

Parameters:
v_i: Value of vehicle type i (file_1_view_0, column Value)
w_i: Weight (inventory space required) of vehicle type i (file_1_view_0, column Weight)
C: total inventory capacity (file_0_view_0, column Capacity)

Decision Variables:
x_i: number of units of vehicle type i to order daily (integer, x_i ≥ 0)

Objective:
maximize  ∑_{i ∈ I} v_i x_i

Subject to:
∑_{i ∈ I} w_i x_i ≤ C

x_i ∈ ℤ_≥0 for all i ∈ I

Data Mapping

Index Sets:
I: file_1_view_0, column ProductName

Parameters:
v_i: file_1_view_0, column Value, key ProductName
w_i: file_1_view_0, column Weight, key ProductName
C: file_0_view_0, column Capacity

Decision Variables:
x_i: number of units of vehicle type i to order daily (integer, x_i ≥ 0), indexed by file_1_view_0, column ProductName

Objective:
maximize  ∑_{i ∈ I} v_i x_i

Constraint:
∑_{i ∈ I} w_i x_i ≤ C

x_i ∈ ℤ_≥0 for all i ∈ I