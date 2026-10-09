Let $x_i$ be the number of vehicles of type $i$ to order per day. Each $x_i$ is a nonnegative integer.

Let $I$ be the set of VehicleIDs from the data.

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the daily inventory limit (Capacity) for vehicle type $i$.

Let $T = \sum_{i \in I} u_i$ be the total inventory capacity per day.

Let $m_i$ be the mapping from VehicleID $i$ to ProductName.

The data is as follows (preserving source order):

| VehicleID | VehicleType         | Capacity ($u_i$) | Value ($b_i$) |
|-----------|---------------------|------------------|---------------|
| 1         | Sedans              | 100              | 1200          |
| 2         | SUVs                | 80               | 1800          |
| 3         | Electric Vehicles   | 120              | 2500          |
| 4         | Hybrid Vehicles     | 90               | 2000          |
| 5         | Trucks              | 50               | 1500          |
| 6         | Sports Cars         | 30               | 3000          |
| 7         | Compact Cars        | 110              | 1000          |
| 8         | Luxury Sedans       | 40               | 3500          |
| 9         | Vans                | 60               | 1600          |
| 10        | Pickup Trucks       | 35               | 1700          |

Total inventory capacity per day:
$$
T = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715
$$

#### Mathematical Model

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\ldots,10\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} b_i x_i
$$

**Subject to:**

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

- $b_1 = 1200$ (Sedans)
- $b_2 = 1800$ (SUVs)
- $b_3 = 2500$ (Electric Vehicles)
- $b_4 = 2000$ (Hybrid Vehicles)
- $b_5 = 1500$ (Trucks)
- $b_6 = 3000$ (Sports Cars)
- $b_7 = 1000$ (Compact Cars)
- $b_8 = 3500$ (Luxury Sedans)
- $b_9 = 1600$ (Vans)
- $b_{10} = 1700$ (Pickup Trucks)

- $u_1 = 100$, $u_2 = 80$, $u_3 = 120$, $u_4 = 90$, $u_5 = 50$, $u_6 = 30$, $u_7 = 110$, $u_8 = 40$, $u_9 = 60$, $u_{10} = 35$

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

**Mapping of $x_i$ to VehicleType:**

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