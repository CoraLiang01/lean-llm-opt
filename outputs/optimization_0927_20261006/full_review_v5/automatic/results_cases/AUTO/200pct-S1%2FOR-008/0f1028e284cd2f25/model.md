Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the following VehicleTypes in source order:

| VehicleID | VehicleType         | Per-Type Capacity ($u_i$) | Benefit Coefficient ($p_i$) |
|-----------|---------------------|---------------------------|-----------------------------|
| 1         | Sedans              | 100                       | 1200                        |
| 2         | SUVs                | 80                        | 1800                        |
| 3         | Electric Vehicles   | 120                       | 2500                        |
| 4         | Hybrid Vehicles     | 90                        | 2000                        |
| 5         | Trucks              | 50                        | 1500                        |
| 6         | Sports Cars         | 30                        | 3000                        |
| 7         | Compact Cars        | 110                       | 1000                        |
| 8         | Luxury Sedans       | 40                        | 3500                        |
| 9         | Vans                | 60                        | 1600                        |
| 10        | Pickup Trucks       | 35                        | 1700                        |

Let $x_i \in \mathbb{Z}_{\geq 0}$ for all $i$.

Let $C = \sum_{i=1}^{10} u_i = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$ be the total inventory capacity per day.

The complete model is:

**Objective:**
\[
\max \quad 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
\]

**Subject to:**

_Per-type daily inventory limits:_
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

_Total daily inventory capacity:_
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715
\]

_Nonnegativity and integrality:_
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
\]

**Parameter Table (source order):**

| VehicleID | VehicleType         | Per-Type Capacity | Benefit Coefficient |
|-----------|---------------------|-------------------|---------------------|
| 1         | Sedans              | 100               | 1200                |
| 2         | SUVs                | 80                | 1800                |
| 3         | Electric Vehicles   | 120               | 2500                |
| 4         | Hybrid Vehicles     | 90                | 2000                |
| 5         | Trucks              | 50                | 1500                |
| 6         | Sports Cars         | 30                | 3000                |
| 7         | Compact Cars        | 110               | 1000                |
| 8         | Luxury Sedans       | 40                | 3500                |
| 9         | Vans                | 60                | 1600                |
| 10        | Pickup Trucks       | 35                | 1700                |