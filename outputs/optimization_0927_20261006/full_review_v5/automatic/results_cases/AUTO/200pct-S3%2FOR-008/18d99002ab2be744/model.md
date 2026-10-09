Let $I$ be the set of vehicle types, indexed by $i$, with VehicleID and VehicleType as identifiers. Let $x_i$ be the integer number of vehicles of type $i$ to order per day.

**Parameters:**

| VehicleID | VehicleType        | Capacity | Value |
|-----------|-------------------|----------|-------|
| 1         | Sedans            | 100      | 1200  |
| 2         | SUVs              | 80       | 1800  |
| 3         | Electric Vehicles | 120      | 2500  |
| 4         | Hybrid Vehicles   | 90       | 2000  |
| 5         | Trucks            | 50       | 1500  |
| 6         | Sports Cars       | 30       | 3000  |
| 7         | Compact Cars      | 110      | 1000  |
| 8         | Luxury Sedans     | 40       | 3500  |
| 9         | Vans              | 60       | 1600  |
| 10        | Pickup Trucks     | 35       | 1700  |

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the per-type daily inventory limit (Capacity) for vehicle type $i$.

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ be the total inventory capacity per day.

---

### Mathematical Model

**Decision variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (number of vehicles of type $i$ to order per day)

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Subject to:**

1. **Per-type daily inventory limits:**
   \[
   x_i \leq u_i, \quad \forall i \in I
   \]

2. **Total inventory capacity:**
   \[
   \sum_{i \in I} x_i \leq 715
   \]

3. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   \]

---

**Where:**

- $I = \{1,2,3,4,5,6,7,8,9,10\}$ (VehicleID)
- $b_i$ and $u_i$ as given in the table above.

---

**Parameter Table (source order):**

| VehicleID | VehicleType        | Capacity ($u_i$) | Value ($b_i$) |
|-----------|-------------------|------------------|---------------|
| 1         | Sedans            | 100              | 1200          |
| 2         | SUVs              | 80               | 1800          |
| 3         | Electric Vehicles | 120              | 2500          |
| 4         | Hybrid Vehicles   | 90               | 2000          |
| 5         | Trucks            | 50               | 1500          |
| 6         | Sports Cars       | 30               | 3000          |
| 7         | Compact Cars      | 110              | 1000          |
| 8         | Luxury Sedans     | 40               | 3500          |
| 9         | Vans              | 60               | 1600          |
| 10        | Pickup Trucks     | 35               | 1700          |