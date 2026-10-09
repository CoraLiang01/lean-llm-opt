Mathematical Model

Index Sets:
- Let I be the set of vehicle types, identified by VehicleType in file_0_view_0 and ProductName in file_1_view_0.

Parameters:
- b_i: benefit coefficient of vehicle type i (from Value in file_1_view_0, matched to ProductName = VehicleType).
- u_i: daily inventory limit for vehicle type i (from Capacity in file_0_view_0, indexed by VehicleType).
- C: total inventory capacity per day, defined as $C = \sum_{i \in I} u_i$.

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Constraints:
1. Vehicle-type daily inventory limits:
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
- $b_i$: Value from file_1_view_0, where ProductName = VehicleType.
- $u_i$: Capacity from file_0_view_0, indexed by VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of Capacity from file_0_view_0).
- $x_i$: Decision variable for each $i \in I$.

All parameters are mapped directly from the CSV files as described above.