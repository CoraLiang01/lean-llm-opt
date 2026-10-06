ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑉: Set of vehicle types, indexed by i. (VehicleType/ProductName from data)

Parameters:
- 𝑐𝑎𝑝𝑖: Daily inventory limit (capacity) for vehicle type i.  
  Data Mapping: file_0_view_0, column Capacity, keyed by VehicleType
- 𝑏𝑒𝑛𝑒𝑓𝑖𝑡𝑖: Benefit coefficient for vehicle type i.  
  Data Mapping: file_1_view_0, column Value, keyed by ProductName

Decision Variables:
- 𝑥𝑖: Number of units of vehicle type i to order per day (integer, 𝑥𝑖 ≥ 0, ∀i ∈ 𝑉)

Objective:
- Maximize total benefit:
  
  maximize ∑_{i ∈ 𝑉} 𝑏𝑒𝑛𝑒𝑓𝑖𝑡𝑖 · 𝑥𝑖

Constraints:
1. Per-vehicle-type daily inventory limits:
  𝑥𝑖 ≤ 𝑐𝑎𝑝𝑖  ∀i ∈ 𝑉

2. Total inventory capacity (sum of all ordered units does not exceed total inventory capacity):
  ∑_{i ∈ 𝑉} 𝑥𝑖 ≤ ∑_{i ∈ 𝑉} 𝑐𝑎𝑝𝑖

3. Integer and nonnegativity:
  𝑥𝑖 ∈ ℤ₊  ∀i ∈ 𝑉

DATA MAPPING

- Index set 𝑉: All VehicleType values from file_0_view_0 and all ProductName values from file_1_view_0 (matched by name).
- Parameter 𝑐𝑎𝑝𝑖: file_0_view_0, column Capacity, keyed by VehicleType.
- Parameter 𝑏𝑒𝑛𝑒𝑓𝑖𝑡𝑖: file_1_view_0, column Value, keyed by ProductName.
- Decision variable 𝑥𝑖: defined for each i ∈ 𝑉.

All mappings use the original row order and supplied file order. No business IDs are synthesized.