Mathematical Model

Index Sets:
- Let I be the set of vehicle types, with each vehicle type identified by VehicleType in file_0_view_0 and ProductName in file_1_view_0.

Parameters:
- $b_i$: Benefit coefficient for vehicle type $i$ (from Value in file_1_view_0, matched to VehicleType).
- $u_i$: Daily inventory limit for vehicle type $i$ (from Capacity in file_0_view_0).
- $C$: Total inventory capacity per day, defined as $C = \sum_{i \in I} u_i$.

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Constraints:
1. Vehicle type daily inventory limits:
$$
x_i \leq u_i \quad \forall i \in I
$$

2. Total inventory capacity:
$$
\sum_{i \in I} x_i \leq C
$$

3. Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping

- $I$: All VehicleType in file_0_view_0 and ProductName in file_1_view_0.
- $b_i$: file_1_view_0, column Value, matched by ProductName = VehicleType.
- $u_i$: file_0_view_0, column Capacity, by VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0, column Capacity).
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are defined using the exact identifiers and columns from the current CSVQA data.