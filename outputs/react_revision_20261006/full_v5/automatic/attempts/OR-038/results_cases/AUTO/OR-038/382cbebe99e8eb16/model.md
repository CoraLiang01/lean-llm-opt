Mathematical Optimization Model

Index Sets:
- $I$: Set of vehicle types, indexed by $i$ (from file_0_view_0.VehicleType and file_1_view_0.ProductName).

Parameters:
- $b_i$: Benefit coefficient of vehicle type $i$ (from file_1_view_0.Value, matched by ProductName = VehicleType).
- $u_i$: Daily inventory limit (maximum units that can be ordered per day) for vehicle type $i$ (from file_0_view_0.Capacity).
- $C$: Total inventory capacity per day, $C = \sum_{i \in I} u_i$.

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day, integer and $x_i \geq 0$.

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
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

Data Mapping:
- $I$: All vehicle types in file_0_view_0.VehicleType and file_1_view_0.ProductName (matched by name).
- $b_i$: file_1_view_0.Value, with $i$ = file_1_view_0.ProductName.
- $u_i$: file_0_view_0.Capacity, with $i$ = file_0_view_0.VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0.Capacity).
- $x_i$: Number of vehicles of type $i$ to order per day (decision variable, integer, for each $i$).

All parameters are mapped directly from the CSV files as described above.