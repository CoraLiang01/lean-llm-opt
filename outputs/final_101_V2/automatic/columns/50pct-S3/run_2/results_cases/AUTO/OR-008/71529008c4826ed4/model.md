Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as given by VehicleID and VehicleType.

Parameters (from the data):

| VehicleID | VehicleType        | Capacity | Value |
|-----------|-------------------|----------|-------|
| 1         | Sedans            | 100      | 1200  |
| 2         | SUVs              | 80       | 1800  |
| 3         | Electric Vehicles | 120      | 2500  |
| 4         | Hybrid Vehicles   | 90       | 2000  |
| 5         | Trucks            | 50       | 1500  |
| 6         | Sports Cars       | 30       | 3000  |
| 7         | Compact Cars      | 110      | 1000  |
| 8         | Luxury Sedans     | 40       | 3500  |
| 9         | Vans              | 60       | 1600  |
| 10        | Pickup Trucks     | 35       | 1700  |

Let $x_i$ be the integer number of vehicles of type $i$ to order per day.

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the per-type daily inventory limit (Capacity) for vehicle type $i$.

Let $U = \sum_{i=1}^{10} u_i$ be the total inventory capacity per day.

The model is:

$$
\text{Maximize} \quad \sum_{i=1}^{10} b_i x_i
$$

Subject to:
\[
\begin{align*}
& x_i \leq u_i, \quad \forall i = 1,\ldots,10 \\
& \sum_{i=1}^{10} x_i \leq U \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\end{align*}
\]

Where:

- $b_1 = 1200$, $b_2 = 1800$, $b_3 = 2500$, $b_4 = 2000$, $b_5 = 1500$, $b_6 = 3000$, $b_7 = 1000$, $b_8 = 3500$, $b_9 = 1600$, $b_{10} = 1700$
- $u_1 = 100$, $u_2 = 80$, $u_3 = 120$, $u_4 = 90$, $u_5 = 50$, $u_6 = 30$, $u_7 = 110$, $u_8 = 40$, $u_9 = 60$, $u_{10} = 35$
- $U = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

Explicitly, the model is:

$$
\text{Maximize} \quad 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
$$

Subject to:
\[
\begin{align*}
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
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\end{align*}
\]

Where $x_1$ = Sedans, $x_2$ = SUVs, $x_3$ = Electric Vehicles, $x_4$ = Hybrid Vehicles, $x_5$ = Trucks, $x_6$ = Sports Cars, $x_7$ = Compact Cars, $x_8$ = Luxury Sedans, $x_9$ = Vans, $x_{10}$ = Pickup Trucks.