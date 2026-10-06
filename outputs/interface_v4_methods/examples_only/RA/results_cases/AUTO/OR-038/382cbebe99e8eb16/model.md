ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑉: Set of vehicle types, indexed by i. (Aligned by VehicleType/ProductName across both tables.)

Parameters:
- b_i: Benefit coefficient for vehicle type i.
  Data Mapping: b_i = Value where ProductName = i from file_1_view_0 (products.csv)
- c_i: Daily inventory limit for vehicle type i.
  Data Mapping: c_i = Capacity where VehicleType = i from file_0_view_0 (capacity.csv)
- C_total: Total inventory capacity per day.
  Data Mapping: C_total = ∑_{i ∈ 𝑉} Capacity from file_0_view_0 (capacity.csv)

Decision Variables:
- x_i: Number of vehicles of type i to order per day (integer, x_i ≥ 0)

Objective:
Maximize total benefit:
  max ∑_{i ∈ 𝑉} b_i x_i

Constraints:
1. Per-vehicle-type daily inventory limit:
  x_i ≤ c_i  ∀ i ∈ 𝑉

2. Total inventory capacity:
  ∑_{i ∈ 𝑉} x_i ≤ C_total

3. Integrality and nonnegativity:
  x_i ∈ ℤ_+,  ∀ i ∈ 𝑉

DATA MAPPING

- Index set 𝑉 is defined by the set of VehicleType in file_0_view_0 (capacity.csv) and ProductName in file_1_view_0 (products.csv), matched by name.
- b_i: file_1_view_0, column Value, key ProductName = i
- c_i: file_0_view_0, column Capacity, key VehicleType = i
- C_total: sum of Capacity over all rows in file_0_view_0

Each x_i is indexed by the explicit business identifier VehicleType/ProductName as present in the source files.