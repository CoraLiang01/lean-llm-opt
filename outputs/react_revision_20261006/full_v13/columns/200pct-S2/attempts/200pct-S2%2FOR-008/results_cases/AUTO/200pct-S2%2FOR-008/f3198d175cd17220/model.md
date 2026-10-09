#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $x_i \geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$
- $u_i$ = daily inventory limit (capacity) for vehicle type $i$

Parameters $b_i$ and $u_i$ are mapped below.

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} x_i \leq \sum_{i \in I} u_i
\]
\[
0 \leq x_i \leq u_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

#### Data Mapping

- $I$: All VehicleType/ProductName values from both tables.
- $b_i$: file_1_view_0[ProductName, Value], where $b_i$ is the Value for ProductName $i$.
- $u_i$: file_0_view_0[VehicleType, Capacity], where $u_i$ is the Capacity for VehicleType $i$.
- The total inventory capacity is $\sum_{i \in I} u_i$ (sum of all per-type capacities).

- Decision variables: $x_i$ for each $i \in I$, integer, $0 \leq x_i \leq u_i$.

- The objective maximizes total benefit from all ordered vehicles, subject to per-type and total inventory limits.