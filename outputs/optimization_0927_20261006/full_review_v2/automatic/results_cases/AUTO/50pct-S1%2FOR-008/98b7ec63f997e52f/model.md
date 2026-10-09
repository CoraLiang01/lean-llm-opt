Let $x_i$ be the number of vehicles of type $i$ to order per day. Each $x_i$ is a nonnegative integer.

Let the set of vehicle types and their corresponding IDs, benefit coefficients, and daily inventory limits be as follows (in source order):

| VehicleID | VehicleType         | Value | Capacity |
|-----------|---------------------|-------|----------|
| 1         | Sedans              | 1200  | 100      |
| 2         | SUVs                | 1800  | 80       |
| 3         | Electric Vehicles   | 2500  | 120      |
| 4         | Hybrid Vehicles     | 2000  | 90       |
| 5         | Trucks              | 1500  | 50       |
| 6         | Sports Cars         | 3000  | 30       |
| 7         | Compact Cars        | 1000  | 110      |
| 8         | Luxury Sedans       | 3500  | 40       |
| 9         | Vans                | 1600  | 60       |
| 10        | Pickup Trucks       | 1700  | 35       |

Let $x_i$ correspond to VehicleID $i$ and VehicleType as above.

Define:
- $v_i$ = Value for vehicle type $i$
- $u_i$ = Capacity for vehicle type $i$

Let $T = \sum_{i=1}^{10} x_i$ (total vehicles ordered per day)

Let $C_{tot} = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

The model is:

**Objective:**
\[
\max \sum_{i=1}^{10} v_i x_i
\]
where
\[
(v_1, v_2, ..., v_{10}) = (1200, 1800, 2500, 2000, 1500, 3000, 1000, 3500, 1600, 1700)
\]

**Subject to:**

1. **Per-vehicle-type daily inventory limits:**
   \[
   0 \leq x_i \leq u_i \qquad \forall i = 1, ..., 10
   \]
   where
   \[
   (u_1, u_2, ..., u_{10}) = (100, 80, 120, 90, 50, 30, 110, 40, 60, 35)
   \]

2. **Total inventory capacity:**
   \[
   \sum_{i=1}^{10} x_i \leq 715
   \]

3. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, ..., 10
   \]

**Decision variables:**
- $x_i$ = number of vehicles of type $i$ to order per day (integer, $0 \leq x_i \leq u_i$)

**Data (in source order):**

- VehicleID 1: Sedans, Value = 1200, Capacity = 100
- VehicleID 2: SUVs, Value = 1800, Capacity = 80
- VehicleID 3: Electric Vehicles, Value = 2500, Capacity = 120
- VehicleID 4: Hybrid Vehicles, Value = 2000, Capacity = 90
- VehicleID 5: Trucks, Value = 1500, Capacity = 50
- VehicleID 6: Sports Cars, Value = 3000, Capacity = 30
- VehicleID 7: Compact Cars, Value = 1000, Capacity = 110
- VehicleID 8: Luxury Sedans, Value = 3500, Capacity = 40
- VehicleID 9: Vans, Value = 1600, Capacity = 60
- VehicleID 10: Pickup Trucks, Value = 1700, Capacity = 35

**Complete Model:**

\[
\begin{align*}
\max\quad & 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10} \\
\text{s.t.}\quad
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
& x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, ..., 10
\end{align*}
\]