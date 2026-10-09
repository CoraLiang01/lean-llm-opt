Mathematical Model

Index Sets:
- $I$: Set of vehicle types, with business identifiers from file_0_view_0.VehicleType and file_1_view_0.ProductName.

Parameters:
- $b_i$: Benefit coefficient of vehicle type $i$ (from file_1_view_0.Value, key: ProductName = VehicleType).
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$ (from file_0_view_0.Capacity, key: VehicleType).
- $C$: Total inventory capacity per day, defined as $\sum_{i \in I} u_i$.

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day. $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$.

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} x_i \leq C
\]
\[
0 \leq x_i \leq u_i, \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

Data Mapping

- $I$: All VehicleType values from file_0_view_0.VehicleType and file_1_view_0.ProductName.
- $b_i$: file_1_view_0.Value, matched by ProductName = VehicleType.
- $u_i$: file_0_view_0.Capacity, matched by VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of all file_0_view_0.Capacity).
- $x_i$: Decision variable for each $i \in I$.

All parameters and index sets are defined by the current rows in the referenced files.