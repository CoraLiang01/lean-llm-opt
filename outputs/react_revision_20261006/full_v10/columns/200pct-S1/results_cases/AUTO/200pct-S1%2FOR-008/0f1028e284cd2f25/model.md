Mathematical Model

Index Sets:
Let I be the set of vehicle types, identified by VehicleType in file_0_view_0.Capacity and file_1_view_0.ProductName.

Parameters:
Let b_i = Value for vehicle type i, from file_1_view_0.Value where file_1_view_0.ProductName = i.
Let u_i = Capacity for vehicle type i, from file_0_view_0.Capacity where file_0_view_0.VehicleType = i.

Decision Variables:
For each vehicle type i ∈ I:
 x_i = number of vehicles of type i to order per day, integer, x_i ≥ 0.

Objective:
Maximize total benefit:
 maximize ∑_{i ∈ I} b_i x_i

Constraints:
1. Vehicle type daily inventory limits:
  x_i ≤ u_i  for all i ∈ I

2. Nonnegativity and integrality:
  x_i ∈ ℤ_{≥0}  for all i ∈ I

Data Mapping

Index Sets:
I: file_0_view_0.VehicleType and file_1_view_0.ProductName (matched by value equality)

Parameters:
b_i: file_1_view_0.Value, indexed by file_1_view_0.ProductName = i
u_i: file_0_view_0.Capacity, indexed by file_0_view_0.VehicleType = i

Decision Variables:
x_i: number of vehicles of type i to order per day (integer, nonnegative), for i ∈ I

Objective:
maximize ∑_{i ∈ I} b_i x_i

Constraints:
x_i ≤ u_i  for all i ∈ I
x_i ∈ ℤ_{≥0}  for all i ∈ I