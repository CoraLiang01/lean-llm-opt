##### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

Parameters:
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Daily inventory limit (capacity) for vehicle type $i$.
- $C$: Total inventory capacity per day.

Decision variables:
- $x_i$: Number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$).

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} x_i \leq C
\]
\[
x_i \leq u_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Where:
- $C = \sum_{i \in I} u_i$ (the sum of all per-type daily inventory limits, as the total inventory capacity).

##### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv), in source order.
- $u_i$: file_0_view_0, column Capacity, keyed by VehicleType.
- $b_i$: file_1_view_0, column Value, keyed by ProductName (matching VehicleType).
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values from file_0_view_0).
- $x_i$: Decision variable for each $i \in I$.

All parameters are mapped directly from the returned tables, preserving original row and identifier order.