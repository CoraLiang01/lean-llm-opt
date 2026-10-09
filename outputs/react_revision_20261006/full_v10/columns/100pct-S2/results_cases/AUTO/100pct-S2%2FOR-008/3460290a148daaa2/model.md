Mathematical Model

Index Sets:
I: set of vehicle types, as identified by VehicleType in file_0_view_0 and ProductName in file_1_view_0

Parameters:
b_i: benefit coefficient for vehicle type i (from Value in file_1_view_0, matched by ProductName = VehicleType)
u_i: daily inventory limit for vehicle type i (from Capacity in file_0_view_0, indexed by VehicleType)
x_i: number of vehicles of type i to order per day (decision variable)

Variables:
x_i ∈ ℤ≥0  for all i ∈ I

Objective:
maximize  ∑_{i∈I} b_i x_i

Constraints:
(1) Vehicle-type daily inventory limits:
  x_i ≤ u_i  for all i ∈ I

(2) Total inventory capacity:
  ∑_{i∈I} x_i ≤ ∑_{i∈I} u_i

(3) Integrality:
  x_i ∈ ℤ≥0  for all i ∈ I

Data Mapping

Index Set I:
  VehicleType from file_0_view_0 (capacity.csv)
  ProductName from file_1_view_0 (products.csv)
  (Match by VehicleType = ProductName)

Parameter b_i:
  Value column from file_1_view_0 (products.csv), indexed by ProductName

Parameter u_i:
  Capacity column from file_0_view_0 (capacity.csv), indexed by VehicleType

Decision variable x_i:
  Defined for each i ∈ I

All parameters and index sets are defined using the exact column names and table_ids as above.