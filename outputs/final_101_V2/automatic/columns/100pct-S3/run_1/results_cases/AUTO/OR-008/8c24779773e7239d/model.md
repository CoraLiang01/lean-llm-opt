Let $i$ index the vehicle types as given in the original file order.

Let $x_i$ = number of units of vehicle type $i$ to order per day (integer, $\geq 0$).

Parameters (from products.csv and capacity.csv, in source order):

| $i$ | ProductName         | Value | Capacity |
|-----|---------------------|-------|----------|
| 1   | Sedans              | 1200  | 100      |
| 2   | SUVs                | 1800  | 80       |
| 3   | Electric Vehicles   | 2500  | 120      |
| 4   | Hybrid Vehicles     | 2000  | 90       |
| 5   | Trucks              | 1500  | 50       |
| 6   | Sports Cars         | 3000  | 30       |
| 7   | Compact Cars        | 1000  | 110      |
| 8   | Luxury Sedans       | 3500  | 40       |
| 9   | Vans                | 1600  | 60       |
| 10  | Pickup Trucks       | 1700  | 35       |

Let $C_i$ = Capacity for vehicle type $i$ (from capacity.csv, column "Capacity").

Let $v_i$ = Value for vehicle type $i$ (from products.csv, column "Value").

Let $T = \sum_{i=1}^{10} C_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ (total inventory capacity per day).

The complete integer optimization model is:

Objective:
$$
\max \sum_{i=1}^{10} v_i x_i
$$

Subject to:
\[
\begin{align*}
& x_i \leq C_i, \quad \forall i = 1,\ldots,10 \\
& \sum_{i=1}^{10} x_i \leq 715 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\end{align*}
\]

Where:

- $v_i$ and $C_i$ are as given above for each vehicle type $i$ in the original file order.
- $x_i$ is the integer number of units of vehicle type $i$ to order per day.

All data and constraints are included as per the user description and retrieved files.