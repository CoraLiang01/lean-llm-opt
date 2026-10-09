Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ is indexed by VehicleID as given below.

#### Sets and Parameters (in source order):

- Vehicle types and their benefit coefficients:

| VehicleID | VehicleType         | Value |
|-----------|---------------------|-------|
| 1         | Sedans              | 1200  |
| 2         | SUVs                | 1800  |
| 3         | Electric Vehicles   | 2500  |
| 4         | Hybrid Vehicles     | 2000  |
| 5         | Trucks              | 1500  |
| 6         | Sports Cars         | 3000  |
| 7         | Compact Cars        | 1000  |
| 8         | Luxury Sedans       | 3500  |
| 9         | Vans                | 1600  |
| 10        | Pickup Trucks       | 1700  |

- Per-vehicle-type daily inventory limits:

| VehicleID | VehicleType         | Capacity |
|-----------|---------------------|----------|
| 1         | Sedans              | 100      |
| 2         | SUVs                | 80       |
| 3         | Electric Vehicles   | 120      |
| 4         | Hybrid Vehicles     | 90       |
| 5         | Trucks              | 50       |
| 6         | Sports Cars         | 30       |
| 7         | Compact Cars        | 110      |
| 8         | Luxury Sedans       | 40       |
| 9         | Vans                | 60       |
| 10        | Pickup Trucks       | 35       |

- Total daily inventory capacity: $C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

#### Decision Variables:

- $x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1, \ldots, 10$ (number of vehicles of type $i$ to order per day)

#### Objective:

$$
\max \quad 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
$$

#### Constraints:

1. **Per-vehicle-type daily inventory limits:**
   $$
   \begin{align*}
   x_1 &\leq 100 \\
   x_2 &\leq 80 \\
   x_3 &\leq 120 \\
   x_4 &\leq 90 \\
   x_5 &\leq 50 \\
   x_6 &\leq 30 \\
   x_7 &\leq 110 \\
   x_8 &\leq 40 \\
   x_9 &\leq 60 \\
   x_{10} &\leq 35 \\
   \end{align*}
   $$

2. **Total daily inventory capacity:**
   $$
   x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715
   $$

3. **Nonnegativity and integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
   $$

#### Complete Model (all data in source order):

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
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\end{align*}
$$

Where the mapping of $i$ to vehicle type is as listed above.