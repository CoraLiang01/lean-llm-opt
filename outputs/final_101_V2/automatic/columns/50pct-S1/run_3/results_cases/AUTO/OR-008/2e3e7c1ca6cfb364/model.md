Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as listed in the data.

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

| VehicleType         | Capacity |
|---------------------|----------|
| Sedans              | 100      |
| SUVs                | 80       |
| Electric Vehicles   | 120      |
| Hybrid Vehicles     | 90       |
| Trucks              | 50       |
| Sports Cars         | 30       |
| Compact Cars        | 110      |
| Luxury Sedans       | 40       |
| Vans                | 60       |
| Pickup Trucks       | 35       |

Let $I$ be the set of all vehicle types as listed above.

Let $v_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the per-vehicle-type daily inventory limit (Capacity) for vehicle type $i$.

Let $C$ be the total inventory capacity per day (sum of all $u_i$):

$$
C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715
$$

**Decision Variables:**

$x_i \in \mathbb{Z}_{\geq 0}$, for each vehicle type $i \in I$.

---

**Mathematical Model:**

**Objective:**
$$
\max \sum_{i \in I} v_i x_i
$$

**Subject to:**

_Per-vehicle-type daily inventory limits:_
$$
x_i \leq u_i, \quad \forall i \in I
$$

_Total inventory capacity:_
$$
\sum_{i \in I} x_i \leq 715
$$

_Nonnegativity and integrality:_
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
$$

---

**Explicitly, with identifiers and coefficients:**

**Objective:**
$$
\max \Big(
1200\,x_{\text{Sedans}} +
1800\,x_{\text{SUVs}} +
2500\,x_{\text{Electric Vehicles}} +
2000\,x_{\text{Hybrid Vehicles}} +
1500\,x_{\text{Trucks}} +
3000\,x_{\text{Sports Cars}} +
1000\,x_{\text{Compact Cars}} +
3500\,x_{\text{Luxury Sedans}} +
1600\,x_{\text{Vans}} +
1700\,x_{\text{Pickup Trucks}}
\Big)
$$

**Subject to:**
\[
\begin{align*}
x_{\text{Sedans}} &\leq 100 \\
x_{\text{SUVs}} &\leq 80 \\
x_{\text{Electric Vehicles}} &\leq 120 \\
x_{\text{Hybrid Vehicles}} &\leq 90 \\
x_{\text{Trucks}} &\leq 50 \\
x_{\text{Sports Cars}} &\leq 30 \\
x_{\text{Compact Cars}} &\leq 110 \\
x_{\text{Luxury Sedans}} &\leq 40 \\
x_{\text{Vans}} &\leq 60 \\
x_{\text{Pickup Trucks}} &\leq 35 \\
x_{\text{Sedans}} + x_{\text{SUVs}} + x_{\text{Electric Vehicles}} + x_{\text{Hybrid Vehicles}} + x_{\text{Trucks}} + x_{\text{Sports Cars}} + x_{\text{Compact Cars}} + x_{\text{Luxury Sedans}} + x_{\text{Vans}} + x_{\text{Pickup Trucks}} &\leq 715 \\
x_i &\in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
\end{align*}
\]

Where $I$ is the set of vehicle types as listed above.