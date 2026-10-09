Mathematical Model

Index Sets:
I: set of vehicle types, as given by file_0_view_0.VehicleType and file_1_view_0.ProductName

Parameters:
b_i: benefit coefficient for vehicle type i (from file_1_view_0.Value, matched where file_1_view_0.ProductName = file_0_view_0.VehicleType)
u_i: daily inventory limit for vehicle type i (from file_0_view_0.Capacity)
x_i: number of vehicles of type i to order per day (decision variable)

Variables:
x_i ∈ ℤ≥0 for all i ∈ I

Objective:
maximize ∑_{i ∈ I} b_i x_i

Constraints:
1. Vehicle-type-specific inventory limits:
  x_i ≤ u_i  for all i ∈ I

2. Total inventory capacity:
  ∑_{i ∈ I} x_i ≤ ∑_{i ∈ I} u_i

3. Integrality:
  x_i ∈ ℤ≥0  for all i ∈ I

Data Mapping

Index Sets:
I = file_0_view_0.VehicleType = file_1_view_0.ProductName

Parameters:
b_i = file_1_view_0.Value, matched by ProductName = VehicleType
u_i = file_0_view_0.Capacity

Variables:
x_i: number of vehicles of type i to order per day

All parameters and index sets are defined by the current rows of the referenced files.