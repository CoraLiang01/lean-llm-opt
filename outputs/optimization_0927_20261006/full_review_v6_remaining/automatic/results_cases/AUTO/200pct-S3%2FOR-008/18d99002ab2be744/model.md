Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID from the data.

Parameters:

- $b_i$: Benefit coefficient for vehicle type $i$ (from "Value" in products.csv, matched to VehicleType/VehicleID).
- $u_i$: Maximum daily inventory for vehicle type $i$ (from "Capacity" in capacity.csv, by VehicleID).
- $C$: Total inventory capacity per day (sum of all $x_i$ must not exceed this).

Data:

| VehicleID | VehicleType        | $u_i$ (Capacity) | $b_i$ (Value) |
|-----------|-------------------|------------------|---------------|
| 1         | Sedans            | 100              | 1200          |
| 2         | SUVs              | 80               | 1800          |
| 3         | Electric Vehicles | 120              | 2500          |
| 4         | Hybrid Vehicles   | 90               | 2000          |
| 5         | Trucks            | 50               | 1500          |
| 6         | Sports Cars       | 30               | 3000          |
| 7         | Compact Cars      | 110              | 1000          |
| 8         | Luxury Sedans     | 40               | 3500          |
| 9         | Vans              | 60               | 1600          |
| 10        | Pickup Trucks     | 35               | 1700          |

Let $C$ be the total inventory capacity per day (this is not explicitly given in the data; if not specified, use $\sum_{i=1}^{10} x_i \leq C$ as the total inventory constraint, with $C$ to be provided by the user or as a parameter).

Model:

Objective:
$$
\max \sum_{i=1}^{10} b_i x_i
$$

Subject to:

Per-vehicle-type daily inventory limits:
$$
x_i \leq u_i \qquad \forall i = 1,\ldots,10
$$

Total inventory capacity:
$$
\sum_{i=1}^{10} x_i \leq C
$$

Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10
$$

Where:

- $b_i$ and $u_i$ are as given in the table above for each VehicleID.
- $C$ is the total inventory capacity per day (to be specified).

Decision variables:

- $x_i$: Number of vehicles of type $i$ to order per day (integer, $\geq 0$).

All identifiers and coefficients are as retrieved from the data.