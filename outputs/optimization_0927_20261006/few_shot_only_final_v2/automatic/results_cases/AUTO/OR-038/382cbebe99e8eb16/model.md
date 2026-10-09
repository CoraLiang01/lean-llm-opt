Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the vehicle types as listed in the data.

**Parameters:**

- From products.csv:
    - $v_i$: Value (benefit) of one unit of vehicle type $i$
    - $w_i$: Weight (resource consumption) of one unit of vehicle type $i$
    - Vehicle types $i$:
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

- From capacity.csv:
    - $c_i$: Capacity (maximum number of units) for vehicle type $i$

**Model:**

Maximize total benefit:
$$
\max \quad 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
$$

Subject to:

Vehicle-type-specific capacity constraints:
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

Non-negativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\]

**Where:**

- $x_1$: Sedans
- $x_2$: SUVs
- $x_3$: Electric Vehicles
- $x_4$: Hybrid Vehicles
- $x_5$: Trucks
- $x_6$: Sports Cars
- $x_7$: Compact Cars
- $x_8$: Luxury Sedans
- $x_9$: Vans
- $x_{10}$: Pickup Trucks

**Complete Model:**

\[
\begin{align*}
\max \quad & 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10} \\
\text{s.t.} \quad 
& x_1 \leq 100 \\
& x_2 \leq 80 \\
& x_3 \leq 120 \\
& x_4 \leq 90 \\
& x_5 \leq 50 \\
& x_6 \leq 30 \\
& x_7 \leq 110 \\
& x_8 \leq 40 \\
& x_9 \leq 60 \\
& x_{10} \leq 35 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 10
\end{align*}
\]