Abstract Mathematical Model

Sets:
- $I$: set of vehicle types, indexed by $i$.

Parameters:
- $b_i$: benefit coefficient for vehicle type $i$ (from products.csv, column Value, key ProductName).
- $u_i$: daily inventory limit for vehicle type $i$ (from capacity.csv, column Capacity, key VehicleType).
- $U$: total inventory capacity per day (sum of all $u_i$).

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$.

Objective:
\[
\max \sum_{i \in I} b_i x_i
\]

Subject to:
\[
\sum_{i \in I} x_i \leq U
\]
\[
x_i \leq u_i \quad \forall i \in I
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

Data Mapping

- $I$: All VehicleType values from file_0_view_0 (capacity.csv, column VehicleType) and all ProductName values from file_1_view_0 (products.csv, column ProductName). Match by VehicleType = ProductName.
- $b_i$: file_1_view_0, column Value, key ProductName.
- $u_i$: file_0_view_0, column Capacity, key VehicleType.
- $U$: $\sum_{i \in I} u_i$ (sum of all file_0_view_0, column Capacity).
- $x_i$: decision variable for each $i \in I$.

All parameters are mapped directly from the returned CSVQA data, using the exact column names and business identifiers.