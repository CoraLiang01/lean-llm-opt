ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $I$: Set of vehicle types, indexed by $i$ (with business keys $(\text{VehicleID}, \text{VehicleType})$ from capacity.csv and $\text{ProductName}$ from products.csv).

Parameters:
- $b_i$: Benefit coefficient for vehicle type $i$.
- $u_i$: Daily inventory limit for vehicle type $i$.
- $C$: Total inventory capacity per day.

Decision Variables:
- $x_i$: Number of vehicles of type $i$ to order per day. ($x_i \in \mathbb{Z}_{\geq 0}$)

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Constraints:
1. Per-vehicle-type daily inventory limits:
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

Data Mapping:

- $b_i$ (benefit coefficient): file_1_view_0[ProductName=$i$].Value
- $u_i$ (per-vehicle-type limit): file_0_view_0[VehicleID,VehicleType=($i$)].Capacity
- $C$ (total inventory capacity): $\sum_{i \in I} u_i$ (sum of file_0_view_0[:].Capacity)
- $x_i$: Decision variable for vehicle type $i$ (indexed by $(\text{VehicleID}, \text{VehicleType})$ and $\text{ProductName}$ as matched between files)

All parameters and indices are to be matched using the business keys as provided in the source files, preserving original row and file order.