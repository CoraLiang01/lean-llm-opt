##### Decision Variables

Let $x_i$ be the integer number of vehicles of type $i$ to order per day, for each $i$ in the set of vehicle types below.

##### Parameters

Let the set of vehicle types $i$ be:

| VehicleID | VehicleType         | Capacity | Value (Benefit Coefficient) | Weight |
|-----------|---------------------|----------|-----------------------------|--------|
| 1         | Sedans              | 100      | 1200                        | 20     |
| 2         | SUVs                | 80       | 1800                        | 15     |
| 3         | Electric Vehicles   | 120      | 2500                        | 25     |
| 4         | Hybrid Vehicles     | 90       | 2000                        | 18     |
| 5         | Trucks              | 50       | 1500                        | 10     |
| 6         | Sports Cars         | 30       | 3000                        | 5      |
| 7         | Compact Cars        | 110      | 1000                        | 22     |
| 8         | Luxury Sedans       | 40       | 3500                        | 8      |
| 9         | Vans                | 60       | 1600                        | 12     |
| 10        | Pickup Trucks       | 35       | 1700                        | 7      |

Let $c_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the inventory limit (Capacity) for vehicle type $i$.

Let $w_i$ be the weight per unit for vehicle type $i$.

Let $W = \sum_{i=1}^{10} u_i \cdot w_i$ be the total inventory capacity (in weight units).

##### Objective Function

$\max \sum_{i=1}^{10} c_i x_i$

##### Constraints

1. Vehicle type inventory limits:
   $$
   0 \leq x_i \leq u_i,\quad x_i \in \mathbb{Z},\quad \forall i=1,\ldots,10
   $$
2. Total inventory capacity constraint:
   $$
   \sum_{i=1}^{10} w_i x_i \leq W
   $$

##### Numerical Model

Let the variables and parameters be indexed as follows:

- $x_1$: Sedans, $u_1=100$, $c_1=1200$, $w_1=20$
- $x_2$: SUVs, $u_2=80$, $c_2=1800$, $w_2=15$
- $x_3$: Electric Vehicles, $u_3=120$, $c_3=2500$, $w_3=25$
- $x_4$: Hybrid Vehicles, $u_4=90$, $c_4=2000$, $w_4=18$
- $x_5$: Trucks, $u_5=50$, $c_5=1500$, $w_5=10$
- $x_6$: Sports Cars, $u_6=30$, $c_6=3000$, $w_6=5$
- $x_7$: Compact Cars, $u_7=110$, $c_7=1000$, $w_7=22$
- $x_8$: Luxury Sedans, $u_8=40$, $c_8=3500$, $w_8=8$
- $x_9$: Vans, $u_9=60$, $c_9=1600$, $w_9=12$
- $x_{10}$: Pickup Trucks, $u_{10}=35$, $c_{10}=1700$, $w_{10}=7$

Objective:
$$
\max\ 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10}
$$

Subject to:
\[
\begin{align*}
0 &\leq x_1 \leq 100 \\
0 &\leq x_2 \leq 80 \\
0 &\leq x_3 \leq 120 \\
0 &\leq x_4 \leq 90 \\
0 &\leq x_5 \leq 50 \\
0 &\leq x_6 \leq 30 \\
0 &\leq x_7 \leq 110 \\
0 &\leq x_8 \leq 40 \\
0 &\leq x_9 \leq 60 \\
0 &\leq x_{10} \leq 35 \\
20x_1 + 15x_2 + 25x_3 + 18x_4 + 10x_5 + 5x_6 + 22x_7 + 8x_8 + 12x_9 + 7x_{10} &\leq W \\
x_i &\in \mathbb{Z},\quad \forall i=1,\ldots,10
\end{align*}
\]

where $W = 100 \times 20 + 80 \times 15 + 120 \times 25 + 90 \times 18 + 50 \times 10 + 30 \times 5 + 110 \times 22 + 40 \times 8 + 60 \times 12 + 35 \times 7 = 2000 + 1200 + 3000 + 1620 + 500 + 150 + 2420 + 320 + 720 + 245 = 11,\!175$.

So,
$$
20x_1 + 15x_2 + 25x_3 + 18x_4 + 10x_5 + 5x_6 + 22x_7 + 8x_8 + 12x_9 + 7x_{10} \leq 11,\!175
$$

and $x_i$ integer for all $i=1,\ldots,10$.