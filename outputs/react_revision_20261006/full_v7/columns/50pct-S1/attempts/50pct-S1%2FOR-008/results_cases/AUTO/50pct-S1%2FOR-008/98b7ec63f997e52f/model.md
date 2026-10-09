#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $b_i$ = benefit coefficient for vehicle type $i$ (from Value in products.csv)
- $u_i$ = daily inventory limit for vehicle type $i$ (from Capacity in capacity.csv)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $x_i \geq 0$)

Define:
- $C = \sum_{i \in I} u_i$ (total inventory capacity per day, as the sum of all per-type limits)

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

- $I$ (vehicle types): All VehicleType from file_0_view_0 (capacity.csv) and all ProductName from file_1_view_0 (products.csv). Matched by name.
- $b_i$: Value column from file_1_view_0 (products.csv), matched to $i$ by ProductName.
- $u_i$: Capacity column from file_0_view_0 (capacity.csv), matched to $i$ by VehicleType.
- $C$: $\sum_{i \in I} u_i$, i.e., sum of all Capacity values in file_0_view_0.
- $x_i$: Decision variable for each $i \in I$.

Each $x_i$ is bounded above by its per-type limit $u_i$, and the sum of all $x_i$ is bounded by the total inventory capacity $C$. All variables are nonnegative integers.