**Sets and Indices:**

- Let $i$ index vehicle types, corresponding to the "ProductName" column in products.csv.

**Parameters:**

From products.csv:
- $v_i$ = Value of vehicle type $i$  
- $w_i$ = Weight of vehicle type $i$

From capacity.csv:
- For each vehicle type $i$, $c_i$ = Capacity for vehicle type $i$

**Decision Variables:**

- $x_i$ = Number of vehicles of type $i$ to order per day, $x_i \in \mathbb{Z}_{\geq 0}$

**Mathematical Model:**

Maximize total benefit:
$$
\max \sum_{i} v_i x_i
$$

Subject to per-vehicle-type capacity constraints:
$$
x_i \leq c_i \quad \forall i
$$

And nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
$$

**Numerical Formulation (using retrieved data):**

Let $i$ index the following vehicle types:

| VehicleID | ProductName         | Value | Weight | Capacity |
|-----------|---------------------|-------|--------|----------|
| 1         | Sedans              | 1200  | 20     | 100      |
| 2         | SUVs                | 1800  | 15     | 80       |
| 3         | Electric Vehicles   | 2500  | 25     | 120      |
| 4         | Hybrid Vehicles     | 2000  | 18     | 90       |
| 5         | Trucks              | 1500  | 10     | 50       |
| 6         | Sports Cars         | 3000  | 5      | 30       |
| 7         | Compact Cars        | 1000  | 22     | 110      |
| 8         | Luxury Sedans       | 3500  | 8      | 40       |
| 9         | Vans                | 1600  | 12     | 60       |
| 10        | Pickup Trucks       | 1700  | 7      | 35       |

Let $x_i$ be the number of vehicles of type $i$ to order per day.

**Objective:**
$$
\max \Big(
1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
\Big)
$$

**Subject to:**
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
x_i &\in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\}
\end{align*}
\]

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