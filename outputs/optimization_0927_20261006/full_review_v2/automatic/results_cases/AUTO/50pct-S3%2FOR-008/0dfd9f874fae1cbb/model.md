Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID and VehicleType as given below.

**Parameters (from source order):**

| VehicleID | VehicleType        | Capacity | Value |
|-----------|--------------------|----------|-------|
| 1         | Sedans             | 100      | 1200  |
| 2         | SUVs               | 80       | 1800  |
| 3         | Electric Vehicles  | 120      | 2500  |
| 4         | Hybrid Vehicles    | 90       | 2000  |
| 5         | Trucks             | 50       | 1500  |
| 6         | Sports Cars        | 30       | 3000  |
| 7         | Compact Cars       | 110      | 1000  |
| 8         | Luxury Sedans      | 40       | 3500  |
| 9         | Vans               | 60       | 1600  |
| 10        | Pickup Trucks      | 35       | 1700  |

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the daily inventory limit (Capacity) for vehicle type $i$.

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity).

---

### Mathematical Model

**Decision variables:**
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
$$

**Objective:**
$$
\max \left( 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10} \right)
$$

**Subject to:**

1. **Per-vehicle-type daily inventory limits:**
   $$
   x_1 \leq 100
   $$
   $$
   x_2 \leq 80
   $$
   $$
   x_3 \leq 120
   $$
   $$
   x_4 \leq 90
   $$
   $$
   x_5 \leq 50
   $$
   $$
   x_6 \leq 30
   $$
   $$
   x_7 \leq 110
   $$
   $$
   x_8 \leq 40
   $$
   $$
   x_9 \leq 60
   $$
   $$
   x_{10} \leq 35
   $$

2. **Total inventory capacity:**
   $$
   x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715
   $$

3. **Integrality and nonnegativity:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
   $$

---

**Where:**

- $x_1$ = Sedans
- $x_2$ = SUVs
- $x_3$ = Electric Vehicles
- $x_4$ = Hybrid Vehicles
- $x_5$ = Trucks
- $x_6$ = Sports Cars
- $x_7$ = Compact Cars
- $x_8$ = Luxury Sedans
- $x_9$ = Vans
- $x_{10}$ = Pickup Trucks

All coefficients and limits are as retrieved and in source order.