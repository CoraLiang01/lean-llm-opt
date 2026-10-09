Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the following VehicleType and VehicleID:

| $i$ | VehicleID | VehicleType         | Capacity | Value |
|-----|-----------|---------------------|----------|-------|
| 1   | 1         | Sedans              | 100      | 1200  |
| 2   | 2         | SUVs                | 80       | 1800  |
| 3   | 3         | Electric Vehicles   | 120      | 2500  |
| 4   | 4         | Hybrid Vehicles     | 90       | 2000  |
| 5   | 5         | Trucks              | 50       | 1500  |
| 6   | 6         | Sports Cars         | 30       | 3000  |
| 7   | 7         | Compact Cars        | 110      | 1000  |
| 8   | 8         | Luxury Sedans       | 40       | 3500  |
| 9   | 9         | Vans                | 60       | 1600  |
| 10  | 10        | Pickup Trucks       | 35       | 1700  |

Let $N = 10$ (number of vehicle types).

Define:
- $x_i$: integer, number of vehicles of type $i$ to order per day ($i = 1, \ldots, 10$)
- $b_i$: benefit coefficient (Value) for vehicle type $i$
- $u_i$: daily inventory limit (Capacity) for vehicle type $i$
- $C$: total inventory capacity per day

##### Objective:
\[
\max \sum_{i=1}^{10} b_i x_i
\]
where
\[
(b_1, \ldots, b_{10}) = (1200, 1800, 2500, 2000, 1500, 3000, 1000, 3500, 1600, 1700)
\]

##### Constraints:

1. **Per-vehicle-type daily inventory limits:**
\[
x_i \leq u_i \qquad \forall i = 1, \ldots, 10
\]
where
\[
(u_1, \ldots, u_{10}) = (100, 80, 120, 90, 50, 30, 110, 40, 60, 35)
\]

2. **Total inventory capacity:**
\[
\sum_{i=1}^{10} x_i \leq C
\]
where
\[
C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715
\]

3. **Nonnegativity and integrality:**
\[
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10
\]

##### Complete Model

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
                  & x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1, \ldots, 10
\end{align*}
\]

##### Data Used (in source order):

From capacity.csv:
- VehicleID: 1, VehicleType: Sedans, Capacity: 100
- VehicleID: 2, VehicleType: SUVs, Capacity: 80
- VehicleID: 3, VehicleType: Electric Vehicles, Capacity: 120
- VehicleID: 4, VehicleType: Hybrid Vehicles, Capacity: 90
- VehicleID: 5, VehicleType: Trucks, Capacity: 50
- VehicleID: 6, VehicleType: Sports Cars, Capacity: 30
- VehicleID: 7, VehicleType: Compact Cars, Capacity: 110
- VehicleID: 8, VehicleType: Luxury Sedans, Capacity: 40
- VehicleID: 9, VehicleType: Vans, Capacity: 60
- VehicleID: 10, VehicleType: Pickup Trucks, Capacity: 35

From products.csv:
- ProductName: Sedans, Value: 1200
- ProductName: SUVs, Value: 1800
- ProductName: Electric Vehicles, Value: 2500
- ProductName: Hybrid Vehicles, Value: 2000
- ProductName: Trucks, Value: 1500
- ProductName: Sports Cars, Value: 3000
- ProductName: Compact Cars, Value: 1000
- ProductName: Luxury Sedans, Value: 3500
- ProductName: Vans, Value: 1600
- ProductName: Pickup Trucks, Value: 1700