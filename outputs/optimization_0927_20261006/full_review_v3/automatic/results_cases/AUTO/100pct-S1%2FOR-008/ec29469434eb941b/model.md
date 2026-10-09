Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following VehicleTypes by their VehicleID:

| VehicleID | VehicleType         | Capacity | Value |
|-----------|---------------------|----------|-------|
| 1         | Sedans              | 100      | 1200  |
| 2         | SUVs                | 80       | 1800  |
| 3         | Electric Vehicles   | 120      | 2500  |
| 4         | Hybrid Vehicles     | 90       | 2000  |
| 5         | Trucks              | 50       | 1500  |
| 6         | Sports Cars         | 30       | 3000  |
| 7         | Compact Cars        | 110      | 1000  |
| 8         | Luxury Sedans       | 40       | 3500  |
| 9         | Vans                | 60       | 1600  |
| 10        | Pickup Trucks       | 35       | 1700  |

Let $b_i$ be the Value for vehicle type $i$ (see table above).

Let $u_i$ be the Capacity for vehicle type $i$ (see table above).

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity per day).

The mathematical model is:

Objective:
\[
\max \sum_{i=1}^{10} b_i x_i
\]

Subject to:
\[
\sum_{i=1}^{10} x_i \leq 715
\]
\[
0 \leq x_i \leq u_i \qquad \forall i = 1,\ldots,10
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10
\]

Where:
- $x_i$ = number of vehicles of type $i$ to order per day
- $b_i$ = benefit coefficient (Value) for vehicle type $i$
- $u_i$ = daily inventory limit (Capacity) for vehicle type $i$
- The sum of all $x_i$ does not exceed the total inventory capacity $715$
- Each $x_i$ does not exceed its type-specific limit $u_i$ and is a nonnegative integer

All identifiers and coefficients are as retrieved and in source order.