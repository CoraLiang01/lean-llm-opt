Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as listed below. All $x_i$ are nonnegative integers.

**Parameters (in source order):**

| VehicleID | VehicleType         | Per-Type Capacity | Benefit Coefficient |
|-----------|--------------------|-------------------|---------------------|
| 1         | Sedans             | 100               | 1200                |
| 2         | SUVs               | 80                | 1800                |
| 3         | Electric Vehicles  | 120               | 2500                |
| 4         | Hybrid Vehicles    | 90                | 2000                |
| 5         | Trucks             | 50                | 1500                |
| 6         | Sports Cars        | 30                | 3000                |
| 7         | Compact Cars       | 110               | 1000                |
| 8         | Luxury Sedans      | 40                | 3500                |
| 9         | Vans               | 60                | 1600                |
| 10        | Pickup Trucks      | 35                | 1700                |

Let $b_i$ be the benefit coefficient for vehicle type $i$ (see table above).

Let $u_i$ be the per-type daily inventory capacity for vehicle type $i$ (see table above).

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity per day).

---

### Mathematical Model

**Decision Variables:**

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\ldots,10\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} b_i x_i
$$

**Subject to:**

1. **Per-Type Inventory Limits:**

$$
x_i \leq u_i, \quad \forall i \in \{1,2,\ldots,10\}
$$

2. **Total Inventory Capacity:**

$$
\sum_{i=1}^{10} x_i \leq 715
$$

3. **Integrality:**

$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,2,\ldots,10\}
$$

---

**Parameter Table (source order):**

| VehicleID | VehicleType         | $b_i$ (Benefit) | $u_i$ (Per-Type Capacity) |
|-----------|--------------------|-----------------|---------------------------|
| 1         | Sedans             | 1200            | 100                       |
| 2         | SUVs               | 1800            | 80                        |
| 3         | Electric Vehicles  | 2500            | 120                       |
| 4         | Hybrid Vehicles    | 2000            | 90                        |
| 5         | Trucks             | 1500            | 50                        |
| 6         | Sports Cars        | 3000            | 30                        |
| 7         | Compact Cars       | 1000            | 110                       |
| 8         | Luxury Sedans      | 3500            | 40                        |
| 9         | Vans               | 1600            | 60                        |
| 10        | Pickup Trucks      | 1700            | 35                        |