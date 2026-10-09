Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID and VehicleType as given below.

**Parameters (in source order):**

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

Let $I = \{1,2,3,4,5,6,7,8,9,10\}$.

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the per-type daily inventory limit (Capacity) for vehicle type $i$.

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity).

---

### Mathematical Model

**Decision variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of vehicles of type $i$ to order per day)

**Objective:**
\[
\max \sum_{i=1}^{10} b_i x_i
\]
where
\[
(b_1, b_2, ..., b_{10}) = (1200, 1800, 2500, 2000, 1500, 3000, 1000, 3500, 1600, 1700)
\]

**Constraints:**

1. **Per-type daily inventory limits:**
   \[
   x_i \leq u_i, \quad \forall i \in I
   \]
   where
   \[
   (u_1, u_2, ..., u_{10}) = (100, 80, 120, 90, 50, 30, 110, 40, 60, 35)
   \]

2. **Total inventory capacity:**
   \[
   \sum_{i=1}^{10} x_i \leq 715
   \]

3. **Integrality and nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

**Full Formulation:**

\[
\begin{align*}
\max \quad & 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10} \\
\text{s.t.} \quad & x_1 \leq 100 \\
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
\]

**Where:**

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

All coefficients and limits are as retrieved, in source order.