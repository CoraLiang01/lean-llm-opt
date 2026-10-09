Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID and VehicleType as given below.

**Indices and Parameters:**

| VehicleID | VehicleType         | Value ($b_i$) | Capacity ($u_i$) |
|-----------|---------------------|--------------|------------------|
| 1         | Sedans              | 1200         | 100              |
| 2         | SUVs                | 1800         | 80               |
| 3         | Electric Vehicles   | 2500         | 120              |
| 4         | Hybrid Vehicles     | 2000         | 90               |
| 5         | Trucks              | 1500         | 50               |
| 6         | Sports Cars         | 3000         | 30               |
| 7         | Compact Cars        | 1000         | 110              |
| 8         | Luxury Sedans       | 3500         | 40               |
| 9         | Vans                | 1600         | 60               |
| 10        | Pickup Trucks       | 1700         | 35               |

Let $N = 10$ (number of vehicle types).

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity per day).

---

**Mathematical Model:**

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1, \ldots, 10$ (number of vehicles of type $i$ to order per day)

**Objective:**
\[
\max \sum_{i=1}^{10} b_i x_i
\]
where $b_i$ is the Value for vehicle type $i$.

**Constraints:**

1. **Per-type daily inventory limits:**
   \[
   x_i \leq u_i \qquad \forall i = 1, \ldots, 10
   \]
   where $u_i$ is the Capacity for vehicle type $i$.

2. **Total inventory capacity:**
   \[
   \sum_{i=1}^{10} x_i \leq 715
   \]

3. **Integrality and nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10
   \]

---

**Full Numerical Formulation:**

\[
\begin{align*}
\max\quad & 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10} \\
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
& x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10
\end{align*}
\]

Where the mapping of $i$ to vehicle type is as listed above.