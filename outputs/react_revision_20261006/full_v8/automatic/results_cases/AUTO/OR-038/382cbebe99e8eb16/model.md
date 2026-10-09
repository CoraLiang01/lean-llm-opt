Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit (capacity) for vehicle type $i$
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

Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv)
- $u_i$: Capacity column from file_0_view_0, keyed by VehicleType
- $b_i$: Value column from file_1_view_0 (products.csv), keyed by ProductName (matching VehicleType)
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values from file_0_view_0)
- $x_i$: Decision variable for each $i \in I$ (vehicle type)