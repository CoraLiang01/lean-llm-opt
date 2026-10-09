#### Sets and Indices

Let $I$ be the set of vehicle types, indexed by $i$, with business IDs and names as below.

#### Parameters

For each $i \in I$:

- $b_i$ = benefit coefficient for vehicle type $i$ (from products.csv, "Value")
- $u_i$ = daily inventory limit for vehicle type $i$ (from capacity.csv, "Capacity")

Let $C = \sum_{i \in I} u_i$ be the total inventory capacity per day (sum of all per-type limits).

#### Decision Variables

For each $i \in I$:

- $x_i$ = number of vehicles of type $i$ to order per day (integer, $0 \leq x_i \leq u_i$)

#### Mathematical Model

**Objective:**
\[
\max \sum_{i \in I} b_i x_i
\]

**Subject to:**

1. **Total Inventory Capacity:**
   \[
   \sum_{i \in I} x_i \leq C
   \]

2. **Per-Type Inventory Limits:**
   \[
   0 \leq x_i \leq u_i \qquad \forall i \in I
   \]
   \[
   x_i \in \mathbb{Z} \qquad \forall i \in I
   \]

---

#### Parameter Tables (source order)

**Vehicle Types, IDs, and Per-Type Limits (from capacity.csv):**

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

**Benefit Coefficients (from products.csv):**

| ProductName        | Value |
|--------------------|-------|
| Sedans             | 1200  |
| SUVs               | 1800  |
| Electric Vehicles  | 2500  |
| Hybrid Vehicles    | 2000  |
| Trucks             | 1500  |
| Sports Cars        | 3000  |
| Compact Cars       | 1000  |
| Luxury Sedans      | 3500  |
| Vans               | 1600  |
| Pickup Trucks      | 1700  |

---

#### Complete Model (with explicit coefficients and IDs):

Let $x_1$ = Sedans, $x_2$ = SUVs, $x_3$ = Electric Vehicles, $x_4$ = Hybrid Vehicles, $x_5$ = Trucks, $x_6$ = Sports Cars, $x_7$ = Compact Cars, $x_8$ = Luxury Sedans, $x_9$ = Vans, $x_{10}$ = Pickup Trucks.

**Objective:**
\[
\max \; 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
\]

**Subject to:**
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 815
\]
\[
0 \leq x_1 \leq 100
\]
\[
0 \leq x_2 \leq 80
\]
\[
0 \leq x_3 \leq 120
\]
\[
0 \leq x_4 \leq 90
\]
\[
0 \leq x_5 \leq 50
\]
\[
0 \leq x_6 \leq 30
\]
\[
0 \leq x_7 \leq 110
\]
\[
0 \leq x_8 \leq 40
\]
\[
0 \leq x_9 \leq 60
\]
\[
0 \leq x_{10} \leq 35
\]
\[
x_i \in \mathbb{Z} \qquad \forall i = 1,\ldots,10
\]

Where the mapping of $x_i$ to vehicle types and IDs is as above.