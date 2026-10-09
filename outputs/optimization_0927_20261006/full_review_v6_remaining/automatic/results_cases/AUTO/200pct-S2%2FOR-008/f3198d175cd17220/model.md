Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID from the data.

Parameters (from products.csv and capacity.csv, in source order):

| VehicleID | VehicleType         | Capacity | Value  |
|-----------|---------------------|----------|--------|
| 1         | Sedans              | 100      | 1200   |
| 2         | SUVs                | 80       | 1800   |
| 3         | Electric Vehicles   | 120      | 2500   |
| 4         | Hybrid Vehicles     | 90       | 2000   |
| 5         | Trucks              | 50       | 1500   |
| 6         | Sports Cars         | 30       | 3000   |
| 7         | Compact Cars        | 110      | 1000   |
| 8         | Luxury Sedans       | 40       | 3500   |
| 9         | Vans                | 60       | 1600   |
| 10        | Pickup Trucks       | 35       | 1700   |

Let $C_i$ be the per-type daily inventory limit (Capacity for VehicleID $i$), and $v_i$ the benefit coefficient (Value for VehicleType $i$).

Let $T = \sum_{i=1}^{10} C_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ be the total inventory capacity per day.

The model:

Maximize total benefit:
$$
\max \sum_{i=1}^{10} v_i x_i
$$

Subject to:

Per-type daily inventory limits:
$$
0 \leq x_i \leq C_i \qquad \forall i = 1,\ldots,10
$$

Total inventory capacity:
$$
\sum_{i=1}^{10} x_i \leq 715
$$

Integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10
$$

Where:

- $v_1 = 1200$, $C_1 = 100$ (Sedans)
- $v_2 = 1800$, $C_2 = 80$ (SUVs)
- $v_3 = 2500$, $C_3 = 120$ (Electric Vehicles)
- $v_4 = 2000$, $C_4 = 90$ (Hybrid Vehicles)
- $v_5 = 1500$, $C_5 = 50$ (Trucks)
- $v_6 = 3000$, $C_6 = 30$ (Sports Cars)
- $v_7 = 1000$, $C_7 = 110$ (Compact Cars)
- $v_8 = 3500$, $C_8 = 40$ (Luxury Sedans)
- $v_9 = 1600$, $C_9 = 60$ (Vans)
- $v_{10} = 1700$, $C_{10} = 35$ (Pickup Trucks)

Decision variables $x_i$ are the number of vehicles of type $i$ to order per day, as integers.