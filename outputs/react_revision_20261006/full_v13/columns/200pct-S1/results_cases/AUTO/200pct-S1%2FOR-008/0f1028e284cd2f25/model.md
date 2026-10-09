#### Mathematical Model

Let:
- $I$ = set of vehicle types, indexed by $i$ (from all VehicleType/ProductName in the data)
- $x_i$ = number of vehicles of type $i$ to order per day (decision variable, integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$ (from products.csv)
- $u_i$ = daily inventory limit for vehicle type $i$ (from capacity.csv)
- $C = \sum_{i \in I} u_i$ = total inventory capacity per day (sum of all vehicle-type capacities)

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Subject to:
1. Vehicle-type daily inventory limits:
$$
0 \leq x_i \leq u_i \qquad \forall i \in I
$$

2. Total inventory capacity:
$$
\sum_{i \in I} x_i \leq C
$$

3. Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

---

#### Data Mapping

- $I$: All VehicleType/ProductName values from both
    - file_0_view_0 (capacity.csv, column VehicleType)
    - file_1_view_0 (products.csv, column ProductName)
    (joined on VehicleType = ProductName)
- $b_i$: file_1_view_0 (products.csv), column Value, for ProductName $i$
- $u_i$: file_0_view_0 (capacity.csv), column Capacity, for VehicleType $i$
- $C$: $\sum_{i \in I} u_i$ (sum of all Capacity values in file_0_view_0)
- $x_i$: Decision variable for each $i \in I$ (vehicle type)

All parameters are mapped directly from the CSV columns as described above.