ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝔓: Set of products (Product), from file_2_view_0["Product"]
- 𝔇: Set of devices (Device), from file_1_view_0["Device"]

Parameters:
- profit_p: Unit profit of product p ∈ 𝔓
    Data Mapping: file_2_view_0["Unit_Profit"], keyed by file_2_view_0["Product"]
- time_dp: Processing time required on device d ∈ 𝔇 per unit of product p ∈ 𝔓
    Data Mapping: file_0_view_0, columns file_0_view_0["Device"] (rows), file_0_view_0["P1"]...["P111"] (columns)
- cap_d: Monthly operating capacity of device d ∈ 𝔇
    Data Mapping: file_1_view_0["Monthly_Capacity"], keyed by file_1_view_0["Device"]

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
- x_p ≥ 0, continuous ∀ p ∈ 𝔓

Data Mapping:
- profit_p: file_2_view_0["Unit_Profit"], keyed by file_2_view_0["Product"]
- time_dp: file_0_view_0, with device d from file_0_view_0["Device"] and product p from file_0_view_0["P1"]...["P111"]
- cap_d: file_1_view_0["Monthly_Capacity"], keyed by file_1_view_0["Device"]