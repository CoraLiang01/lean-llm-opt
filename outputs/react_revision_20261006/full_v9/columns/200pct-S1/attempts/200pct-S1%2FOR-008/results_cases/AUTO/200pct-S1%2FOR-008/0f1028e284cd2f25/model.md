Mathematical Model

Index Sets:
- $I$: Set of vehicle types, identified by VehicleType in file_0_view_0 and ProductName in file_1_view_0.

Parameters:
- $b_i$: Benefit coefficient of vehicle type $i$, from Value in file_1_view_0 where ProductName = VehicleType.
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$, from Capacity in file_0_view_0.

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
x_i \leq u_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All VehicleType in file_0_view_0 and ProductName in file_1_view_0.
- $b_i$: file_1_view_0[ProductName = $i$].Value
- $u_i$: file_0_view_0[VehicleType = $i$].Capacity
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable, integer, nonnegative)