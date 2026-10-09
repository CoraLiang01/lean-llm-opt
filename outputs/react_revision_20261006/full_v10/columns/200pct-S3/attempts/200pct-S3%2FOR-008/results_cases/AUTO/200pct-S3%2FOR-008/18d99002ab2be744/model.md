Mathematical Model

Index Sets:
Let I be the set of vehicle types, identified by VehicleType in file_0_view_0 and ProductName in file_1_view_0.

Parameters:
Let b_i = Value for vehicle type i (from file_1_view_0, ProductName = i).
Let u_i = Capacity for vehicle type i (from file_0_view_0, VehicleType = i).
Let C = ∑_{i∈I} u_i (the total inventory capacity per day).

Decision Variables:
x_i ∈ ℤ₊, for all i ∈ I: number of vehicles of type i to order per day.

Objective:
maximize ∑_{i∈I} b_i x_i

Constraints:
1. Vehicle-type daily inventory limits:
  x_i ≤ u_i  for all i ∈ I

2. Total inventory capacity:
  ∑_{i∈I} x_i ≤ C

3. Integrality and nonnegativity:
  x_i ∈ ℤ₊  for all i ∈ I

Data Mapping

Index Sets:
I: file_0_view_0.VehicleType = file_1_view_0.ProductName

Parameters:
b_i: file_1_view_0.Value, indexed by ProductName
u_i: file_0_view_0.Capacity, indexed by VehicleType
C: sum over file_0_view_0.Capacity

Decision Variables:
x_i: number of vehicles of type i to order per day, indexed by I

Objective:
maximize ∑_{i∈I} file_1_view_0.Value × x_i

Constraints:
x_i ≤ file_0_view_0.Capacity  for all i ∈ I
∑_{i∈I} x_i ≤ sum over file_0_view_0.Capacity
x_i ∈ ℤ₊  for all i ∈ I