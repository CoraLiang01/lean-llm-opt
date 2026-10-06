##### Sets and Indices

Let $I$ be the set of vehicle types, indexed by $i$.

##### Parameters (from retrieved data)

For each vehicle type $i$:

- $b_i$: benefit coefficient (Value from products.csv)
- $u_i$: daily inventory limit for vehicle type $i$ (Capacity from capacity.csv)

Total inventory capacity per day: $C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

| VehicleID | VehicleType        | $u_i$ (Capacity) | $b_i$ (Value) |
|-----------|-------------------|------------------|---------------|
| 1         | Sedans            | 100              | 1200          |
| 2         | SUVs              | 80               | 1800          |
| 3         | Electric Vehicles | 120              | 2500          |
| 4         | Hybrid Vehicles   | 90               | 2000          |
| 5         | Trucks            | 50               | 1500          |
| 6         | Sports Cars       | 30               | 3000          |
| 7         | Compact Cars      | 110              | 1000          |
| 8         | Luxury Sedans     | 40               | 3500          |
| 9         | Vans              | 60               | 1600          |
| 10        | Pickup Trucks     | 35               | 1700          |

##### Decision Variables

For each vehicle type $i$:

- $x_i$: number of vehicles of type $i$ to order per day ($x_i \in \mathbb{Z}_{\geq 0}$)

##### Mathematical Model

Objective:
$$
\max \sum_{i \in I} b_i x_i
$$

Subject to:

1. Per-vehicle-type daily inventory limits:
$$
x_i \leq u_i \qquad \forall i \in I
$$

2. Total inventory capacity constraint:
$$
\sum_{i \in I} x_i \leq 715
$$

3. Integrality and nonnegativity:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
$$

##### Data Used

Vehicle types and parameters (in source order):

- Sedans: $b_1 = 1200$, $u_1 = 100$
- SUVs: $b_2 = 1800$, $u_2 = 80$
- Electric Vehicles: $b_3 = 2500$, $u_3 = 120$
- Hybrid Vehicles: $b_4 = 2000$, $u_4 = 90$
- Trucks: $b_5 = 1500$, $u_5 = 50$
- Sports Cars: $b_6 = 3000$, $u_6 = 30$
- Compact Cars: $b_7 = 1000$, $u_7 = 110$
- Luxury Sedans: $b_8 = 3500$, $u_8 = 40$
- Vans: $b_9 = 1600$, $u_9 = 60$
- Pickup Trucks: $b_{10} = 1700$, $u_{10} = 35$

Total inventory capacity per day: $715$ units.

##### Complete Model

$$
\begin{align*}
\max\quad & 1200x_1 + 1800x_2 + 2500x_3 + 2000x_4 + 1500x_5 + 3000x_6 + 1000x_7 + 3500x_8 + 1600x_9 + 1700x_{10} \\
\text{s.t.}\quad
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
& x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 715 \\
& x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10
\end{align*}
$$

Where the mapping of $x_i$ to vehicle types is as above, in the original source order.