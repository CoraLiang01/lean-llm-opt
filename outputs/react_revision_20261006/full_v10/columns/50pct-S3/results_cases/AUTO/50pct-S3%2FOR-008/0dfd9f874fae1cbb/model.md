Mathematical Model

Index Sets:
- Let I be the set of vehicle types, identified by VehicleType in file_0_view_0 and ProductName in file_1_view_0.

Parameters:
- $b_i$: benefit coefficient of vehicle type $i$ (Value from file_1_view_0, ProductName = VehicleType)
- $u_i$: daily inventory limit for vehicle type $i$ (Capacity from file_0_view_0, VehicleType)
- $C$: total inventory capacity per day, defined as $C = \sum_{i \in I} u_i$

Decision Variables:
- $x_i$: number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$

Objective:
Maximize total benefit:
$$
\max \sum_{i \in I} b_i x_i
$$

Constraints:
1. Vehicle-type daily inventory limits:
$$
x_i \leq u_i \quad \forall i \in I
$$

2. Total inventory capacity:
$$
\sum_{i \in I} x_i \leq C
$$

3. Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
$$

Data Mapping

- Index set $I$:
  - file_0_view_0.VehicleType
  - file_1_view_0.ProductName (matched to VehicleType)
- Parameter $b_i$: file_1_view_0.Value, matched by ProductName = VehicleType
- Parameter $u_i$: file_0_view_0.Capacity, matched by VehicleType
- Total capacity $C$: $\sum_{i \in I} u_i$
- Decision variable $x_i$: number of vehicles of type $i$ to order per day

All data is mapped directly from the returned CSV files, preserving original identifiers.