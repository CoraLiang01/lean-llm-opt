Mathematical Model

Index Sets:
- $I$: Set of vehicle types, indexed by $i$. Each $i$ corresponds to a unique VehicleID from file_0_view_0.

Parameters:
- $b_i$: Benefit coefficient for vehicle type $i$. (from file_1_view_0, column Value, matched by VehicleType/ProductName)
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$. (from file_0_view_0, column Capacity)
- $C$: Total inventory capacity per day, $C = \sum_{i \in I} u_i$

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day. $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} x_i \leq C
\]
\[
0 \leq x_i \leq u_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All VehicleID in file_0_view_0 (capacity.csv), column VehicleID
- $b_i$: file_1_view_0 (products.csv), column Value, matched where file_1_view_0.ProductName = file_0_view_0.VehicleType
- $u_i$: file_0_view_0 (capacity.csv), column Capacity
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0.Capacity)
- $x_i$: Decision variable for each $i \in I$ (VehicleID from file_0_view_0)

All parameters and index sets are defined using the exact columns and business identifiers from the returned CSV data.