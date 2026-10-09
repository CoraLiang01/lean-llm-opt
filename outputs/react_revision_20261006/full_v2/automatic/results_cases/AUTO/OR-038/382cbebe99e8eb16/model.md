##### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

Parameters:
- $b_i$: benefit coefficient for vehicle type $i$
- $u_i$: daily inventory limit for vehicle type $i$

Decision variables:
- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
x_i \leq u_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

##### Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv)
- $u_i$: Capacity column from file_0_view_0, matched by VehicleType
- $b_i$: Value column from file_1_view_0 (products.csv), matched by ProductName = VehicleType
- $x_i$: Decision variable for each $i \in I$