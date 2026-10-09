Mathematical Model

Index Sets:
Let $I$ be the set of vehicle types, identified by VehicleType in file_0_view_0.Capacity and file_1_view_0.ProductName.

Parameters:
For each $i \in I$:
- $b_i$: benefit coefficient of vehicle type $i$ (file_1_view_0, column Value, key ProductName)
- $u_i$: daily inventory limit for vehicle type $i$ (file_0_view_0, column Capacity, key VehicleType)

Decision Variables:
For each $i \in I$:
- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$\max \sum_{i \in I} b_i x_i$

Constraints:
1. Vehicle type daily inventory limits:
$\quad x_i \leq u_i \quad \forall i \in I$

2. Nonnegativity and integrality:
$\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

Data Mapping

- $I$: All VehicleType in file_0_view_0.Capacity and ProductName in file_1_view_0.ProductName
- $b_i$: file_1_view_0.Value, keyed by ProductName
- $u_i$: file_0_view_0.Capacity, keyed by VehicleType
- $x_i$: decision variable for each $i \in I$ (vehicle type)

No total inventory capacity constraint is present in the current data; only per-vehicle-type limits are enforced.