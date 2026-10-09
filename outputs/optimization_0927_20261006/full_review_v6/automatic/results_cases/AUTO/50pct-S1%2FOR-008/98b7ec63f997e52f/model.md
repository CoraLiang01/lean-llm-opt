Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the VehicleID from the data below. All $x_i$ are nonnegative integers.

**Parameters (from retrieved data, in source order):**

| VehicleID | VehicleType         | Per-Type Capacity | Benefit Coefficient |
|-----------|--------------------|-------------------|---------------------|
| 1         | Sedans             | 100               | 1200                |
| 2         | SUVs               | 80                | 1800                |
| 3         | Electric Vehicles  | 120               | 2500                |
| 4         | Hybrid Vehicles    | 90                | 2000                |
| 5         | Trucks             | 50                | 1500                |
| 6         | Sports Cars        | 30                | 3000                |
| 7         | Compact Cars       | 110               | 1000                |
| 8         | Luxury Sedans      | 40                | 3500                |
| 9         | Vans               | 60                | 1600                |
| 10        | Pickup Trucks      | 35                | 1700                |

Let $b_i$ be the benefit coefficient for vehicle type $i$ (from the table above).

Let $u_i$ be the per-type daily capacity for vehicle type $i$ (from the table above).

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity per day).

---

**Mathematical Model:**

Maximize total benefit:
$$
\max \sum_{i=1}^{10} b_i x_i
$$

Subject to:

_Per-type daily inventory limits:_
$$
x_i \leq u_i \qquad \forall i = 1, \ldots, 10
$$

_Total daily inventory capacity:_
$$
\sum_{i=1}^{10} x_i \leq 715
$$

_Nonnegativity and integrality:_
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10
$$

---

**Where:**

- $x_i$ = number of vehicles of type $i$ to order per day (integer, $\geq 0$)
- $b_i$ = benefit coefficient for vehicle type $i$ (see table)
- $u_i$ = per-type daily capacity for vehicle type $i$ (see table)
- $C = 715$ = total daily inventory capacity (sum of all $u_i$)

All data and constraints are included as retrieved and in source order.