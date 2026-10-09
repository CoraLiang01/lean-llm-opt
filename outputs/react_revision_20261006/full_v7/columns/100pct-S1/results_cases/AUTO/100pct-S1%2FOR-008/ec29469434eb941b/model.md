Mathematical Model

Index Sets:
- $I$: Set of vehicle types, as defined by all VehicleType/ProductName in the data.

Parameters:
- $b_i$: Benefit coefficient for vehicle type $i$. (from file_1_view_0, column Value, matched by ProductName = VehicleType)
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$. (from file_0_view_0, column Capacity)
- $C$: Total inventory capacity per day, $C = \sum_{i \in I} u_i$

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

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

Data Mapping

- $I$: All VehicleType in file_0_view_0 (capacity.csv) and all ProductName in file_1_view_0 (products.csv)
- $b_i$: file_1_view_0, column Value, matched by ProductName = VehicleType
- $u_i$: file_0_view_0, column Capacity, by VehicleType
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0, column Capacity)
- $x_i$: Decision variable for each $i \in I$