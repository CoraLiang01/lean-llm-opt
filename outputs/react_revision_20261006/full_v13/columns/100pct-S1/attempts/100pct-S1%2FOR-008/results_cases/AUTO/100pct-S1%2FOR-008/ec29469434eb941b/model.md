#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (with business identifier VehicleID from file_0_view_0)
- $x_i$ = number of vehicles of type $i$ to order per day (integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$ (from file_1_view_0, matched by VehicleType/ProductName)
- $u_i$ = daily inventory limit for vehicle type $i$ (from file_0_view_0, Capacity)
- $C = \sum_{i \in I} u_i$ = total inventory capacity per day

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

#### Data Mapping

- $I$: All VehicleID in file_0_view_0 (capacity.csv)
- $b_i$: file_1_view_0.Value where file_1_view_0.ProductName = file_0_view_0.VehicleType
- $u_i$: file_0_view_0.Capacity
- $C$: $\sum_{i \in I} u_i$ (sum of file_0_view_0.Capacity)
- $x_i$: number of vehicles of type $i$ to order per day (decision variable, indexed by VehicleID)

All parameters and index sets are mapped directly from the current CSVQA data as described above.