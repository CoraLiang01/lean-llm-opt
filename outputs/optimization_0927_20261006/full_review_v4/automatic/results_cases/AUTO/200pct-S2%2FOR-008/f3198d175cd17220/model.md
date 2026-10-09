Let $x_i$ be the number of vehicles of type $i$ to order per day. Each $x_i$ is a nonnegative integer.

Let $I$ be the set of vehicle types, indexed by VehicleID as below.

Let $b_i$ be the benefit coefficient for vehicle type $i$.

Let $u_i$ be the daily inventory limit (capacity) for vehicle type $i$.

Let $C$ be the total inventory capacity per day (sum of all $x_i$ cannot exceed $C$).

#### Data

| VehicleID | VehicleType         | $b_i$ (Value) | $u_i$ (Capacity) |
|-----------|---------------------|---------------|------------------|
| 1         | Sedans              | 1200          | 100              |
| 2         | SUVs                | 1800          | 80               |
| 3         | Electric Vehicles   | 2500          | 120              |
| 4         | Hybrid Vehicles     | 2000          | 90               |
| 5         | Trucks              | 1500          | 50               |
| 6         | Sports Cars         | 3000          | 30               |
| 7         | Compact Cars        | 1000          | 110              |
| 8         | Luxury Sedans       | 3500          | 40               |
| 9         | Vans                | 1600          | 60               |
| 10        | Pickup Trucks       | 1700          | 35               |

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$.

#### Mathematical Model

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\ldots,10\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} b_i x_i
$$

**Constraints:**

1. **Per-vehicle-type daily inventory limits:**
   $$
   x_i \leq u_i, \quad \forall i \in \{1,2,\ldots,10\}
   $$

2. **Total inventory capacity:**
   $$
   \sum_{i=1}^{10} x_i \leq 715
   $$

3. **Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\ldots,10\}
   $$

**Where:**

- $b_1 = 1200$, $u_1 = 100$ (Sedans)
- $b_2 = 1800$, $u_2 = 80$ (SUVs)
- $b_3 = 2500$, $u_3 = 120$ (Electric Vehicles)
- $b_4 = 2000$, $u_4 = 90$ (Hybrid Vehicles)
- $b_5 = 1500$, $u_5 = 50$ (Trucks)
- $b_6 = 3000$, $u_6 = 30$ (Sports Cars)
- $b_7 = 1000$, $u_7 = 110$ (Compact Cars)
- $b_8 = 3500$, $u_8 = 40$ (Luxury Sedans)
- $b_9 = 1600$, $u_9 = 60$ (Vans)
- $b_{10} = 1700$, $u_{10} = 35$ (Pickup Trucks)

**Complete Model:**

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
& x_i \in \mathbb{Z}_{\geq 0}, \quad i=1,\ldots,10
\end{align*}
$$

**Variable mapping:**

- $x_1$: Sedans
- $x_2$: SUVs
- $x_3$: Electric Vehicles
- $x_4$: Hybrid Vehicles
- $x_5$: Trucks
- $x_6$: Sports Cars
- $x_7$: Compact Cars
- $x_8$: Luxury Sedans
- $x_9$: Vans
- $x_{10}$: Pickup Trucks