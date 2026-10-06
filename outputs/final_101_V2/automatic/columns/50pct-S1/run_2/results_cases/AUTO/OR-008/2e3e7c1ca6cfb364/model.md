Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ runs over the following vehicle types, with their benefit coefficients and daily inventory limits as given below.

**Parameters (from products.csv and capacity.csv, in source order):**

| VehicleType         | Value | Capacity |
|---------------------|-------|----------|
| Sedans              | 1200  | 100      |
| SUVs                | 1800  | 80       |
| Electric Vehicles   | 2500  | 120      |
| Hybrid Vehicles     | 2000  | 90       |
| Trucks              | 1500  | 50       |
| Sports Cars         | 3000  | 30       |
| Compact Cars        | 1000  | 110      |
| Luxury Sedans       | 3500  | 40       |
| Vans                | 1600  | 60       |
| Pickup Trucks       | 1700  | 35       |

Let $I$ be the set of all vehicle types above, in the order shown.

Let $v_i$ = Value for vehicle type $i$ (from products.csv).

Let $u_i$ = Capacity for vehicle type $i$ (from capacity.csv).

Let $C = \sum_{i \in I} u_i$ (total inventory capacity per day).

**Decision variables:**

$x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of vehicles of type $i$ to order per day).

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**

1. **Per-vehicle-type daily inventory limits:**
   \[
   x_i \leq u_i, \quad \forall i \in I
   \]

2. **Total inventory capacity:**
   \[
   \sum_{i \in I} x_i \leq C
   \]
   where $C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

3. **Integrality and nonnegativity:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

**Explicit Data (in source order):**

- $I = \{$Sedans, SUVs, Electric Vehicles, Hybrid Vehicles, Trucks, Sports Cars, Compact Cars, Luxury Sedans, Vans, Pickup Trucks$\}$
- $(v_i) = (1200, 1800, 2500, 2000, 1500, 3000, 1000, 3500, 1600, 1700)$
- $(u_i) = (100, 80, 120, 90, 50, 30, 110, 40, 60, 35)$
- $C = 715$

---

**Full Model:**

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
& x_i \in \mathbb{Z}_{\geq 0},\quad i=1,\ldots,10
\end{align*}
\]

where the mapping of $i$ to vehicle type is as listed above.