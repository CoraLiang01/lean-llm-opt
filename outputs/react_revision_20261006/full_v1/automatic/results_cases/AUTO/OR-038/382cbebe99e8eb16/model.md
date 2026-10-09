#### Symbolic Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit for vehicle type $i$
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)

Parameters:
- $b_i$ from products.csv (Value column, keyed by ProductName)
- $u_i$ from capacity.csv (Capacity column, keyed by VehicleType)

Decision variables:
- $x_i \in \mathbb{Z}_{\geq 0}$, $\forall i \in I$

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
x_i \leq u_i, \quad \forall i \in I
\]

\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\]

#### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv), in source order.
- $b_i$: file_1_view_0 (products.csv), Value column, keyed by ProductName = VehicleType.
- $u_i$: file_0_view_0 (capacity.csv), Capacity column, keyed by VehicleType.
- $x_i$: Decision variable for each $i \in I$.

No total inventory capacity constraint is specified in the data; only per-vehicle-type daily limits are enforced. All variables and parameters are mapped directly from the supplied files and columns.