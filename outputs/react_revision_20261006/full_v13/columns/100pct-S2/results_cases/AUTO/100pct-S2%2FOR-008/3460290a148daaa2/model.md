## Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit (capacity) for vehicle type $i$
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable)

### Objective
\[
\max \sum_{i \in I} b_i x_i
\]

### Constraints
1. Vehicle-type-specific daily inventory limits:
   \[
   0 \leq x_i \leq u_i \qquad \forall i \in I
   \]
2. Integrality:
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv)
- $u_i$: Capacity column from file_0_view_0, matched by VehicleType
- $b_i$: Value column from file_1_view_0 (products.csv), matched by ProductName = VehicleType
- $x_i$: Decision variable for each $i \in I$ (vehicle type)

No total inventory capacity constraint is present in the current data; only per-vehicle-type limits are enforced.