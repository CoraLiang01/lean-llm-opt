Mathematical Model

Sets:
- $I$: set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)

Parameters:
- $b_i$: benefit coefficient for vehicle type $i$ (from Value in products.csv, matched by ProductName = VehicleType)
- $u_i$: daily inventory limit for vehicle type $i$ (from Capacity in capacity.csv, indexed by VehicleType)
- $C$: total inventory capacity per day, $C = \sum_{i \in I} u_i$

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day, integer and $x_i \geq 0$

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

- $I$: All VehicleType in file_0_view_0 (capacity.csv) and ProductName in file_1_view_0 (products.csv)
- $b_i$: file_1_view_0, column Value, matched by ProductName = VehicleType
- $u_i$: file_0_view_0, column Capacity, indexed by VehicleType
- $C$: $\sum_{i \in I} u_i$ (sum of Capacity from file_0_view_0)
- $x_i$: decision variable for each $i \in I$