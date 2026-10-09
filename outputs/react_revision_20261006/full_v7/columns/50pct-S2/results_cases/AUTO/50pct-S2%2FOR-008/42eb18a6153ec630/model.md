#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $b_i$ = benefit coefficient for vehicle type $i$ (from Value in products.csv)
- $u_i$ = daily inventory limit for vehicle type $i$ (from Capacity in capacity.csv)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)

Let $C = \sum_{i \in I} u_i$ (total inventory capacity per day, as the sum of all per-type limits).

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

---

#### Data Mapping

- $I$: All vehicle types from file_0_view_0.VehicleType and file_1_view_0.ProductName (matched by name)
- $b_i$: file_1_view_0.Value, with key file_1_view_0.ProductName = $i$
- $u_i$: file_0_view_0.Capacity, with key file_0_view_0.VehicleType = $i$
- $C$: $\sum_{i \in I} u_i$ (sum over all file_0_view_0.Capacity)
- $x_i$: Decision variable for each $i \in I$ (vehicle type)

All parameters are mapped directly from the returned CSV data, preserving original identifiers.