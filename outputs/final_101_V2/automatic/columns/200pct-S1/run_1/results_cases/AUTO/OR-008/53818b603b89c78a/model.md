Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the vehicle types as listed in the data.

**Parameters:**

From products.csv:

| ProductName         | Value |
|---------------------|-------|
| Sedans              | 1200  |
| SUVs                | 1800  |
| Electric Vehicles   | 2500  |
| Hybrid Vehicles     | 2000  |
| Trucks              | 1500  |
| Sports Cars         | 3000  |
| Compact Cars        | 1000  |
| Luxury Sedans       | 3500  |
| Vans                | 1600  |
| Pickup Trucks       | 1700  |

From capacity.csv:

| VehicleID | VehicleType        | Capacity |
|-----------|-------------------|----------|
| 1         | Sedans            | 100      |
| 2         | SUVs              | 80       |
| 3         | Electric Vehicles | 120      |
| 4         | Hybrid Vehicles   | 90       |
| 5         | Trucks            | 50       |
| 6         | Sports Cars       | 30       |
| 7         | Compact Cars      | 110      |
| 8         | Luxury Sedans     | 40       |
| 9         | Vans              | 60       |
| 10        | Pickup Trucks     | 35       |

Let $I$ be the set of all vehicle types as above.

Let $v_i$ be the Value for vehicle type $i$.

Let $u_i$ be the per-type Capacity for vehicle type $i$.

Let $C = \sum_{i \in I} u_i$ be the total inventory capacity per day.

**Decision Variables:**

$x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of vehicles of type $i$ to order per day)

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**

_Per-type inventory limits:_
\[
x_i \leq u_i \qquad \forall i \in I
\]

_Total inventory capacity:_
\[
\sum_{i \in I} x_i \leq C
\]

_Integrality:_
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

**Numerical Data:**

- $I = \{$Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks$\}$
- $(v_i) = $ [1200, 1800, 2500, 2000, 1500, 3000, 1000, 3500, 1600, 1700]
- $(u_i) = $ [100, 80, 120, 90, 50, 30, 110, 40, 60, 35]
- $C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

---

**Explicitly:**

\[
\max \big(1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}\big)
\]

Subject to:

\[
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
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} &\leq 715 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10
\end{align*}
\]

Where the mapping of $i$ to vehicle type is as per the order above.