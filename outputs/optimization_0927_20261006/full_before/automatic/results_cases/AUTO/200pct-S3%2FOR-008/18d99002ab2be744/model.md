Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as given by VehicleID and ProductName.

Parameters (from the data):

- Let $V_i$ = Value of vehicle type $i$ (from "products.csv", column "Value")
- Let $C_i$ = Capacity of vehicle type $i$ (from "capacity.csv", column "Capacity")
- Let $T = $ total inventory capacity $= \sum_{i=1}^{10} C_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

Vehicle types and their parameters (in source order):

| $i$ | VehicleID | ProductName         | $V_i$ | $C_i$ |
|-----|-----------|---------------------|-------|-------|
| 1   | 1         | Sedans              | 1200  | 100   |
| 2   | 2         | SUVs                | 1800  | 80    |
| 3   | 3         | Electric Vehicles   | 2500  | 120   |
| 4   | 4         | Hybrid Vehicles     | 2000  | 90    |
| 5   | 5         | Trucks              | 1500  | 50    |
| 6   | 6         | Sports Cars         | 3000  | 30    |
| 7   | 7         | Compact Cars        | 1000  | 110   |
| 8   | 8         | Luxury Sedans       | 3500  | 40    |
| 9   | 9         | Vans                | 1600  | 60    |
| 10  | 10        | Pickup Trucks       | 1700  | 35    |

Mathematical Model:

Objective:
\[
\max \quad 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
\]

Subject to:

Per-vehicle-type daily inventory limits:
\[
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
\end{align*}
\]

Total inventory capacity constraint:
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715
\]

Integrality and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\]

Where:

- $x_1$ = number of Sedans to order per day
- $x_2$ = number of SUVs to order per day
- $x_3$ = number of Electric Vehicles to order per day
- $x_4$ = number of Hybrid Vehicles to order per day
- $x_5$ = number of Trucks to order per day
- $x_6$ = number of Sports Cars to order per day
- $x_7$ = number of Compact Cars to order per day
- $x_8$ = number of Luxury Sedans to order per day
- $x_9$ = number of Vans to order per day
- $x_{10}$ = number of Pickup Trucks to order per day

All coefficients and identifiers are as retrieved and in source order.