#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (with business identifier VehicleType from file_0_view_0 and ProductName from file_1_view_0)
- $x_i$ = number of vehicles of type $i$ to order per day (integer decision variable)
- $b_i$ = benefit coefficient for vehicle type $i$ (parameter from file_1_view_0, column Value)
- $u_i$ = daily inventory limit for vehicle type $i$ (parameter from file_0_view_0, column Capacity)

Define total inventory capacity as $C = \sum_{i \in I} u_i$.

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

- Index set $I$: All VehicleType values from file_0_view_0 (capacity.csv), matched to ProductName in file_1_view_0 (products.csv).
- $b_i$: file_1_view_0, column Value, keyed by ProductName.
- $u_i$: file_0_view_0, column Capacity, keyed by VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values in file_0_view_0).
- $x_i$: integer variable, for each $i \in I$ (VehicleType/ProductName).

All parameters and index sets are defined by the current rows in the returned tables. No data is omitted or synthesized.