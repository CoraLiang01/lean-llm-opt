Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleType and ProductName as matched below.

Objective:
$$
\max \sum_{i} v_i x_i
$$
where $v_i$ is the Value (benefit coefficient) for vehicle type $i$.

Subject to:

1. Per-vehicle-type daily inventory limits:
$$
x_i \leq c_i \quad \forall i
$$
where $c_i$ is the Capacity for vehicle type $i$.

2. Total inventory capacity constraint:
$$
\sum_{i} x_i \leq \sum_{i} c_i
$$

3. Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

---

#### Data

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

---

#### Complete Model

Let $x_1$ = Sedans, $x_2$ = SUVs, $x_3$ = Electric Vehicles, $x_4$ = Hybrid Vehicles, $x_5$ = Trucks, $x_6$ = Sports Cars, $x_7$ = Compact Cars, $x_8$ = Luxury Sedans, $x_9$ = Vans, $x_{10}$ = Pickup Trucks.

Objective:
$$
\max \ 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
$$

Subject to:
\begin{align*}
x_1 &\leq 100 \\
x_2 &\leq 80 \\
x_3 &\leq 120 \\
x_4 &\leq 90 \\
x_5 &\leq 50 \\
x_6 &\leq 30 \\
x_7 &\leq 110 \\
x_8 &\leq 40 \\
x_9 &\leq 60 \\
x_{10} &\leq 35 \\
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} &\leq 715 \\
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10
\end{align*}

Where the total inventory capacity is $100+80+120+90+50+30+110+40+60+35=715$.