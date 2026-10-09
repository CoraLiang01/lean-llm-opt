### Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$.

Parameters:
- $b_i$: benefit coefficient for vehicle type $i$ (from file_1_view_0, column Value, key ProductName)
- $u_i$: daily inventory limit for vehicle type $i$ (from file_0_view_0, column Capacity, key VehicleType)

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

### Data Mapping

- $I$: All VehicleType values in file_0_view_0 (capacity.csv, column VehicleType)
- $b_i$: file_1_view_0 (products.csv), column Value, key ProductName = $i$
- $u_i$: file_0_view_0 (capacity.csv), column Capacity, key VehicleType = $i$
- $x_i$: integer variable for each $i \in I$