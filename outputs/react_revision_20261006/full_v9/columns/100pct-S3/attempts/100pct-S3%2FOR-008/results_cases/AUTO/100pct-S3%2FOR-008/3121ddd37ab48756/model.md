Mathematical Model

Index Sets:
- $I$: Set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data).

Parameters:
- $b_i$: Benefit coefficient for vehicle type $i$ (from Value in file_1_view_0, matched by ProductName = VehicleType).
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$ (from Capacity in file_0_view_0, matched by VehicleType).
- $C$: Total inventory capacity per day, defined as $C = \sum_{i \in I} u_i$.

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day, integer and $0 \leq x_i \leq u_i$.

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} x_i \leq C
\]
\[
0 \leq x_i \leq u_i \qquad \forall i \in I
\]
\[
x_i \in \mathbb{Z} \qquad \forall i \in I
\]

Data Mapping

- $I$: All VehicleType in file_0_view_0 and ProductName in file_1_view_0.
- $b_i$: file_1_view_0, column Value, with $i$ = ProductName.
- $u_i$: file_0_view_0, column Capacity, with $i$ = VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity in file_0_view_0).
- $x_i$: Decision variable for each $i \in I$.

All parameters are mapped directly from the CSV files as described.