ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑉: Set of vehicle types, indexed by i. (𝑉 = all VehicleType/ProductName in the data)

Parameters:
- 𝑏ᵢ: Benefit coefficient for vehicle type i. (from Value in products.csv)
- 𝑐ᵢ: Daily inventory limit for vehicle type i. (from Capacity in capacity.csv)
- 𝐶_total: Total daily inventory capacity, defined as 𝐶_total = ∑_{i∈𝑉} cᵢ

Decision Variables:
- xᵢ: Number of units of vehicle type i to order per day (integer, xᵢ ≥ 0)

Objective:
Maximize total benefit:
  max ∑_{i∈𝑉} bᵢ xᵢ

Subject to:
1. Per-vehicle-type daily inventory limits:
  xᵢ ≤ cᵢ  ∀ i ∈ 𝑉

2. Total daily inventory capacity:
  ∑_{i∈𝑉} xᵢ ≤ 𝐶_total

3. Integer and nonnegativity:
  xᵢ ∈ ℤ₊  ∀ i ∈ 𝑉

DATA MAPPING

Index Set 𝑉:
- All VehicleType values in file_0_view_0 and all ProductName values in file_1_view_0 (matched by name).

Parameters:
- bᵢ: file_1_view_0, column Value, for ProductName = i
- cᵢ: file_0_view_0, column Capacity, for VehicleType = i
- 𝐶_total: sum of file_0_view_0, column Capacity

Decision Variables:
- xᵢ: Number of units to order for vehicle type i ∈ 𝑉

Notes:
- VehicleType in file_0_view_0 and ProductName in file_1_view_0 are matched by exact string equality.
- All variables are integer and nonnegative.
- All constraints and parameters are mapped directly from the supplied data.