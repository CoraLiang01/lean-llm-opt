Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as given by VehicleID in source order.

Parameters (from retrieved data):

- Vehicle types and IDs:
    1. Sedans
    2. SUVs
    3. Electric Vehicles
    4. Hybrid Vehicles
    5. Trucks
    6. Sports Cars
    7. Compact Cars
    8. Luxury Sedans
    9. Vans
    10. Pickup Trucks

- Benefit coefficients (Value):
    - Sedans: 1200
    - SUVs: 1800
    - Electric Vehicles: 2500
    - Hybrid Vehicles: 2000
    - Trucks: 1500
    - Sports Cars: 3000
    - Compact Cars: 1000
    - Luxury Sedans: 3500
    - Vans: 1600
    - Pickup Trucks: 1700

- Per-type daily inventory limits (Capacity):
    - Sedans: 100
    - SUVs: 80
    - Electric Vehicles: 120
    - Hybrid Vehicles: 90
    - Trucks: 50
    - Sports Cars: 30
    - Compact Cars: 110
    - Luxury Sedans: 40
    - Vans: 60
    - Pickup Trucks: 35

Define $x_i \in \mathbb{Z}_{\geq 0}$ for $i=1,\ldots,10$.

Objective:
$$
\max\ 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
$$

Subject to:

Per-type daily inventory limits:
\begin{align*}
x_1 &\leq 100 \quad \text{(Sedans)} \\
x_2 &\leq 80 \quad \text{(SUVs)} \\
x_3 &\leq 120 \quad \text{(Electric Vehicles)} \\
x_4 &\leq 90 \quad \text{(Hybrid Vehicles)} \\
x_5 &\leq 50 \quad \text{(Trucks)} \\
x_6 &\leq 30 \quad \text{(Sports Cars)} \\
x_7 &\leq 110 \quad \text{(Compact Cars)} \\
x_8 &\leq 40 \quad \text{(Luxury Sedans)} \\
x_9 &\leq 60 \quad \text{(Vans)} \\
x_{10} &\leq 35 \quad \text{(Pickup Trucks)}
\end{align*}

Total inventory capacity constraint:
$$
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq  (100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35)
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10
$$

Where:
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