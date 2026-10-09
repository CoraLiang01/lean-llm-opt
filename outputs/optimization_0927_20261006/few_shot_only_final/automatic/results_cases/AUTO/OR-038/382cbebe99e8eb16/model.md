Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as listed in the tables below.

**Objective:**
\[
\max \sum_{i} v_i x_i
\]
where $v_i$ is the Value for vehicle type $i$.

**Constraints:**

1. **Total Inventory Capacity Constraint:**
\[
\sum_{i} x_i \leq \sum_{i} C_i
\]
where $C_i$ is the Capacity for vehicle type $i$.

2. **Individual Vehicle Type Capacity Constraints:**
\[
x_i \leq C_i \quad \forall i
\]

3. **Non-negativity and Integrality:**
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\]

**Parameters (from source data):**

| VehicleID | VehicleType         | Capacity | Value | Weight |
|-----------|---------------------|----------|-------|--------|
| 1         | Sedans              | 100      | 1200  | 20     |
| 2         | SUVs                | 80       | 1800  | 15     |
| 3         | Electric Vehicles   | 120      | 2500  | 25     |
| 4         | Hybrid Vehicles     | 90       | 2000  | 18     |
| 5         | Trucks              | 50       | 1500  | 10     |
| 6         | Sports Cars         | 30       | 3000  | 5      |
| 7         | Compact Cars        | 110      | 1000  | 22     |
| 8         | Luxury Sedans       | 40       | 3500  | 8      |
| 9         | Vans                | 60       | 1600  | 12     |
| 10        | Pickup Trucks       | 35       | 1700  | 7      |

**Decision Variables:**
\[
x_1 = \text{number of Sedans to order per day}
\]
\[
x_2 = \text{number of SUVs to order per day}
\]
\[
x_3 = \text{number of Electric Vehicles to order per day}
\]
\[
x_4 = \text{number of Hybrid Vehicles to order per day}
\]
\[
x_5 = \text{number of Trucks to order per day}
\]
\[
x_6 = \text{number of Sports Cars to order per day}
\]
\[
x_7 = \text{number of Compact Cars to order per day}
\]
\[
x_8 = \text{number of Luxury Sedans to order per day}
\]
\[
x_9 = \text{number of Vans to order per day}
\]
\[
x_{10} = \text{number of Pickup Trucks to order per day}
\]

**Complete Model:**

\[
\max \; 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
\]

Subject to:
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715
\]
\[
x_1 \leq 100
\]
\[
x_2 \leq 80
\]
\[
x_3 \leq 120
\]
\[
x_4 \leq 90
\]
\[
x_5 \leq 50
\]
\[
x_6 \leq 30
\]
\[
x_7 \leq 110
\]
\[
x_8 \leq 40
\]
\[
x_9 \leq 60
\]
\[
x_{10} \leq 35
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10
\]