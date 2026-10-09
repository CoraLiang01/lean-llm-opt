Let $x_i$ be the number of vehicles of type $i$ to order per day. Each $x_i$ is a nonnegative integer.

Let the set of vehicle types and their corresponding IDs, daily per-type inventory limits, and benefit coefficients be as follows (in source order):

| VehicleID | VehicleType         | Capacity | Value |
|-----------|---------------------|----------|-------|
| 1         | Sedans              | 100      | 1200  |
| 2         | SUVs                | 80       | 1800  |
| 3         | Electric Vehicles   | 120      | 2500  |
| 4         | Hybrid Vehicles     | 90       | 2000  |
| 5         | Trucks              | 50       | 1500  |
| 6         | Sports Cars         | 30       | 3000  |
| 7         | Compact Cars        | 110      | 1000  |
| 8         | Luxury Sedans       | 40       | 3500  |
| 9         | Vans                | 60       | 1600  |
| 10        | Pickup Trucks       | 35       | 1700  |

Let $x_i$ correspond to VehicleID $i$ and VehicleType as above.

Define $C_i$ as the per-type daily inventory limit (Capacity column), and $v_i$ as the benefit coefficient (Value column).

Let $T = \sum_{i=1}^{10} C_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ be the total inventory capacity per day.

The complete integer optimization model is:

Objective:
\[
\max \quad 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
\]

Subject to:
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
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} &\leq 715 \\
x_i &\in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\end{align*}
\]

Where:
- $x_i$ = number of vehicles of type $i$ to order per day (integer, $\geq 0$)
- $i$ indexes VehicleID in the table above
- Each $x_i$ is bounded above by its per-type daily inventory limit (Capacity)
- The sum of all $x_i$ does not exceed the total inventory capacity per day (715)
- The objective maximizes total benefit across all vehicle types

All data, identifiers, and coefficients are preserved in source order as retrieved.