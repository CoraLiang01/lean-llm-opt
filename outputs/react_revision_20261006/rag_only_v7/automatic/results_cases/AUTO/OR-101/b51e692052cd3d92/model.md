Mathematical Optimization Model

Index Sets:
- 𝔓: Set of products (Product in file_2_view_0)
- 𝔇: Set of devices (Device in file_1_view_0)

Parameters:
- profit_p: Unit profit of product p ∈ 𝔓
  Data Mapping: file_2_view_0, column Unit_Profit, key Product
- time_dp: Processing time required for one unit of product p on device d (d ∈ 𝔇, p ∈ 𝔓)
  Data Mapping: file_0_view_0, row Device, column P*, keys Device and Product
- cap_d: Monthly operating capacity of device d ∈ 𝔇
  Data Mapping: file_1_view_0, column Monthly_Capacity, key Device

Decision Variables:
- x_p ≥ 0: Continuous production quantity of product p ∈ 𝔓

Objective:
Maximize total monthly profit:
\[
\max \sum_{p \in 𝔓} \text{profit}_p \cdot x_p
\]

Constraints:
- Device capacity constraints (for each device d ∈ 𝔇):
\[
\sum_{p \in 𝔓} \text{time}_{dp} \cdot x_p \leq \text{cap}_d
\]

Variable Domains:
- x_p ≥ 0, ∀ p ∈ 𝔓 (continuous, nonnegative)

Data Mapping:
- profit_p: file_2_view_0, column Unit_Profit, key Product
- time_dp: file_0_view_0, row Device, column P*, keys Device and Product
- cap_d: file_1_view_0, column Monthly_Capacity, key Device

Index Set Definitions:
- 𝔓 = {Product: all rows in file_2_view_0}
- 𝔇 = {Device: all rows in file_1_view_0}