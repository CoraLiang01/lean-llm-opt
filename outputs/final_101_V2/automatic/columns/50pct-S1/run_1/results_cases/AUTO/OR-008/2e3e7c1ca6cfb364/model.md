Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleType and ProductName as matched below.

**Parameters:**

| $i$ | VehicleType/ProductName      | Value ($b_i$) | Capacity ($u_i$) |
|-----|-----------------------------|--------------|------------------|
| 1   | Sedans                      | 1200         | 100              |
| 2   | SUVs                        | 1800         | 80               |
| 3   | Electric Vehicles           | 2500         | 120              |
| 4   | Hybrid Vehicles             | 2000         | 90               |
| 5   | Trucks                      | 1500         | 50               |
| 6   | Sports Cars                 | 3000         | 30               |
| 7   | Compact Cars                | 1000         | 110              |
| 8   | Luxury Sedans               | 3500         | 40               |
| 9   | Vans                        | 1600         | 60               |
| 10  | Pickup Trucks               | 1700         | 35               |

Let $N = 10$ (number of vehicle types).

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity per day).

---

**Mathematical Model:**

Maximize total benefit:
$$
\max \sum_{i=1}^{10} b_i x_i
$$

Subject to:

1. Per-vehicle-type daily inventory limits:
$$
0 \leq x_i \leq u_i \qquad \forall i = 1, \ldots, 10
$$

2. Total inventory capacity:
$$
\sum_{i=1}^{10} x_i \leq 715
$$

3. Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10
$$

---

**Where:**

- $x_i$ = number of vehicles of type $i$ to order per day
- $b_i$ = benefit coefficient for vehicle type $i$ (see table above)
- $u_i$ = daily inventory limit for vehicle type $i$ (see table above)
- $C = 715$ = total daily inventory capacity

All data and identifiers are preserved in source order.