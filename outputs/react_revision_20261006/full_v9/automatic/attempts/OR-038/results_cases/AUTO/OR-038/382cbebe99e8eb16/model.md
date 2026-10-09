Mathematical Model

Sets:
- $I$: set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)

Parameters:
- $b_i$: benefit coefficient for vehicle type $i$ (from Value in products.csv, matched by ProductName = VehicleType)
- $u_i$: daily inventory limit for vehicle type $i$ (from Capacity in capacity.csv, indexed by VehicleType)
- $C$: total inventory capacity per day (sum of all $u_i$)

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

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

- $I$: All VehicleType/ProductName values from file_0_view_0 (capacity.csv) and file_1_view_0 (products.csv), matched by name.
- $b_i$: file_1_view_0, column Value, with $i$ = ProductName.
- $u_i$: file_0_view_0, column Capacity, with $i$ = VehicleType.
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values in file_0_view_0).
- $x_i$: decision variable for each $i \in I$.

All parameters are mapped directly from the CSV files using the exact column and table_id names. Each vehicle type's benefit and capacity are matched by name. The total inventory capacity $C$ is the sum of all daily inventory limits. The model maximizes total benefit, subject to the total and per-type inventory limits, with integer decision variables.