Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID and VehicleType as given below.

**Parameters:**

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

Let $x_i \in \mathbb{Z}_{\geq 0}$ for all $i=1,\ldots,10$.

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the daily inventory limit (Capacity) for vehicle type $i$.

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity per day).

---

**Mathematical Model:**

Maximize total benefit:
$$
\max \sum_{i=1}^{10} b_i x_i
$$

Subject to:
\[
\begin{align*}
& \sum_{i=1}^{10} x_i \leq 715 \\
& 0 \leq x_1 \leq 100 \\
& 0 \leq x_2 \leq 80 \\
& 0 \leq x_3 \leq 120 \\
& 0 \leq x_4 \leq 90 \\
& 0 \leq x_5 \leq 50 \\
& 0 \leq x_6 \leq 30 \\
& 0 \leq x_7 \leq 110 \\
& 0 \leq x_8 \leq 40 \\
& 0 \leq x_9 \leq 60 \\
& 0 \leq x_{10} \leq 35 \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10
\end{align*}
\]

Where:

- $b_1 = 1200$, $b_2 = 1800$, $b_3 = 2500$, $b_4 = 2000$, $b_5 = 1500$, $b_6 = 3000$, $b_7 = 1000$, $b_8 = 3500$, $b_9 = 1600$, $b_{10} = 1700$
- $u_1 = 100$, $u_2 = 80$, $u_3 = 120$, $u_4 = 90$, $u_5 = 50$, $u_6 = 30$, $u_7 = 110$, $u_8 = 40$, $u_9 = 60$, $u_{10} = 35$

**Decision variables:**

- $x_i$ = number of vehicles of type $i$ to order per day (integer, $0 \leq x_i \leq u_i$)

**Objective:**

- Maximize total benefit from daily vehicle orders, subject to per-type and total inventory limits.