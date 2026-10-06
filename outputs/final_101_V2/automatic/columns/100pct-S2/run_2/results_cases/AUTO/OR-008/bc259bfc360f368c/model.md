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

Let $I$ be the set of all vehicle types as listed above.

Let $v_i$ be the Value for vehicle type $i$.

Let $u_i$ be the per-type capacity for vehicle type $i$.

Let $C = \sum_{i \in I} u_i$ be the total inventory capacity per day.

**Decision Variables:**

$x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day.

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
where $C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

_Nonnegativity and integrality:_
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\]

---

**Explicit Data:**

- $I = \{$Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks$\}$
- $v_i$ and $u_i$ as per the tables above.
- $C = 715$

---

**Full Model:**

\[
\begin{align*}
\max\quad & 1200\,x_{\text{Sedans}} + 1800\,x_{\text{SUVs}} + 2500\,x_{\text{Electric Vehicles}} + 2000\,x_{\text{Hybrid Vehicles}} \\
         & + 1500\,x_{\text{Trucks}} + 3000\,x_{\text{Sports Cars}} + 1000\,x_{\text{Compact Cars}} + 3500\,x_{\text{Luxury Sedans}} \\
         & + 1600\,x_{\text{Vans}} + 1700\,x_{\text{Pickup Trucks}} \\
\text{s.t.}\quad
    & x_{\text{Sedans}} \leq 100 \\
    & x_{\text{SUVs}} \leq 80 \\
    & x_{\text{Electric Vehicles}} \leq 120 \\
    & x_{\text{Hybrid Vehicles}} \leq 90 \\
    & x_{\text{Trucks}} \leq 50 \\
    & x_{\text{Sports Cars}} \leq 30 \\
    & x_{\text{Compact Cars}} \leq 110 \\
    & x_{\text{Luxury Sedans}} \leq 40 \\
    & x_{\text{Vans}} \leq 60 \\
    & x_{\text{Pickup Trucks}} \leq 35 \\
    & x_{\text{Sedans}} + x_{\text{SUVs}} + x_{\text{Electric Vehicles}} + x_{\text{Hybrid Vehicles}} + x_{\text{Trucks}} \\
    & \quad + x_{\text{Sports Cars}} + x_{\text{Compact Cars}} + x_{\text{Luxury Sedans}} + x_{\text{Vans}} + x_{\text{Pickup Trucks}} \leq 715 \\
    & x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
\end{align*}
\]