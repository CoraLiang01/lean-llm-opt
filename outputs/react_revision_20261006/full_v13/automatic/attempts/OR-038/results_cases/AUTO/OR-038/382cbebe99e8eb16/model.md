Mathematical Model

Let $I$ be the set of vehicle types, indexed by $i$, with business identifier VehicleID from file_0_view_0.

Parameters:
- $b_i$: benefit coefficient for vehicle type $i$ (from file_1_view_0, Value, matched to VehicleType/ProductName)
- $u_i$: daily inventory limit for vehicle type $i$ (from file_0_view_0, Capacity)
- $x_i$: number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)

Variables:
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

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

Data Mapping

- $I$: All VehicleID in file_0_view_0 (capacity.csv)
- $b_i$: file_1_view_0.Value, matched where file_0_view_0.VehicleType = file_1_view_0.ProductName
- $u_i$: file_0_view_0.Capacity
- $x_i$: number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)