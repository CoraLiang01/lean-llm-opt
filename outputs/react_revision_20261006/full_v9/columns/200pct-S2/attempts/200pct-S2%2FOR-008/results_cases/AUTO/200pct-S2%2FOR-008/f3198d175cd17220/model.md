Mathematical Model

Index Sets:
- $I$: set of vehicle types, indexed by $i$ (from file_0_view_0.VehicleType and file_1_view_0.ProductName, matched by name)

Parameters:
- $b_i$: benefit coefficient of vehicle type $i$ (from file_1_view_0.Value, key ProductName)
- $u_i$: daily inventory limit (capacity) for vehicle type $i$ (from file_0_view_0.Capacity, key VehicleType)
- $C$: total inventory capacity per day $= \sum_{i \in I} u_i$

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$

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

- $I$: All vehicle types in file_0_view_0.VehicleType and file_1_view_0.ProductName (matched by name)
- $b_i$: file_1_view_0.Value, key ProductName
- $u_i$: file_0_view_0.Capacity, key VehicleType
- $C$: $\sum_{i \in I} u_i$
- $x_i$: number of vehicles of type $i$ to order per day (decision variable, integer, for each $i \in I$)