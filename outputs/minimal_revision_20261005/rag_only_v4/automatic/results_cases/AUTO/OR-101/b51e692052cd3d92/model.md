ABSTRACT MATHEMATICAL MODEL

Sets:
- 𝔓: Set of products (indexed by p), from file_2_view_0[Product]
- 𝔇: Set of devices (indexed by d), from file_1_view_0[Device]

Parameters:
- profit_p: Unit profit of product p  
  Data: file_2_view_0[Unit_Profit] for each p ∈ 𝔓
- time_dp: Processing time required on device d per unit of product p  
  Data: file_0_view_0, row Device d, column p
- cap_d: Monthly operating capacity of device d  
  Data: file_1_view_0[Monthly_Capacity] for each d ∈ 𝔇

Decision Variables:
- x_p ≥ 0: Continuous production quantity of product p (for all p ∈ 𝔓)

Objective:
- Maximize total monthly profit:
  maximize ∑_{p ∈ 𝔓} profit_p · x_p

Constraints:
- Device capacity for each device d ∈ 𝔇:
  ∑_{p ∈ 𝔓} time_dp · x_p ≤ cap_d

Data Mapping

- Set 𝔓: All Product values in file_2_view_0[Product]
- Set 𝔇: All Device values in file_1_view_0[Device]
- profit_p: file_2_view_0[Unit_Profit], indexed by file_2_view_0[Product]
- time_dp: file_0_view_0, row Device (file_0_view_0[Device]), column Product (file_0_view_0 columns P1–P111)
- cap_d: file_1_view_0[Monthly_Capacity], indexed by file_1_view_0[Device]
- Decision variable x_p: defined for all p ∈ 𝔓

All indices, parameters, and constraints are mapped directly to the supplied data tables and columns as described.