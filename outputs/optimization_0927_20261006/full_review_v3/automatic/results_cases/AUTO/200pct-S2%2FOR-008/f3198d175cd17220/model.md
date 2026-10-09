Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID and VehicleType as given below.

#### Sets and Parameters (in source order):

| VehicleID | VehicleType         | Per-Type Capacity | Value (Benefit Coefficient) |
|-----------|---------------------|-------------------|-----------------------------|
| 1         | Sedans              | 100               | 1200                        |
| 2         | SUVs                | 80                | 1800                        |
| 3         | Electric Vehicles   | 120               | 2500                        |
| 4         | Hybrid Vehicles     | 90                | 2000                        |
| 5         | Trucks              | 50                | 1500                        |
| 6         | Sports Cars         | 30                | 3000                        |
| 7         | Compact Cars        | 110               | 1000                        |
| 8         | Luxury Sedans       | 40                | 3500                        |
| 9         | Vans                | 60                | 1600                        |
| 10        | Pickup Trucks       | 35                | 1700                        |

Let $I$ be the set of all VehicleIDs above.

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the per-type daily inventory limit (Capacity) for vehicle type $i$.

Let $C = \sum_{i \in I} u_i$ be the total inventory capacity per day.

#### Decision Variables:

$x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day, for all $i \in I$.

#### Objective:

$$
\max \sum_{i \in I} b_i x_i
$$

#### Constraints:

1. **Per-Type Inventory Limits:**

$$
x_i \leq u_i, \quad \forall i \in I
$$

2. **Total Inventory Capacity:**

$$
\sum_{i \in I} x_i \leq C
$$

3. **Integrality and Nonnegativity:**

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

#### Data Used (in source order):

- VehicleID 1: Sedans, Capacity 100, Value 1200
- VehicleID 2: SUVs, Capacity 80, Value 1800
- VehicleID 3: Electric Vehicles, Capacity 120, Value 2500
- VehicleID 4: Hybrid Vehicles, Capacity 90, Value 2000
- VehicleID 5: Trucks, Capacity 50, Value 1500
- VehicleID 6: Sports Cars, Capacity 30, Value 3000
- VehicleID 7: Compact Cars, Capacity 110, Value 1000
- VehicleID 8: Luxury Sedans, Capacity 40, Value 3500
- VehicleID 9: Vans, Capacity 60, Value 1600
- VehicleID 10: Pickup Trucks, Capacity 35, Value 1700

#### Complete Model:

$$
\begin{align*}
\max \quad & 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10} \\
\text{s.t.} \quad
& x_1 \leq 100 \\
& x_2 \leq 80 \\
& x_3 \leq 120 \\
& x_4 \leq 90 \\
& x_5 \leq 50 \\
& x_6 \leq 30 \\
& x_7 \leq 110 \\
& x_8 \leq 40 \\
& x_9 \leq 60 \\
& x_{10} \leq 35 \\
& x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad i = 1, \ldots, 10
\end{align*}
$$

where the mapping of $x_i$ to vehicle types is as above.