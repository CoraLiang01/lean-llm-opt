Let $x_i$ be the number of vehicles of type $i$ to order per day. Each $x_i$ is a nonnegative integer.

Let $I$ be the set of vehicle types, indexed by VehicleID as in the data below.

Let $b_i$ be the benefit coefficient (Value) for vehicle type $i$.

Let $u_i$ be the daily inventory limit (Capacity) for vehicle type $i$.

Let $x_i$ correspond to VehicleType $i$ as matched by VehicleID/ProductName.

Let $C_{tot} = \sum_{i} u_i$ be the total inventory capacity per day (sum of all per-type capacities).

#### Sets and Parameters

| VehicleID | VehicleType         | Per-Type Capacity $u_i$ | Benefit Coefficient $b_i$ |
|-----------|---------------------|-------------------------|---------------------------|
| 1         | Sedans              | 100                     | 1200                      |
| 2         | SUVs                | 80                      | 1800                      |
| 3         | Electric Vehicles   | 120                     | 2500                      |
| 4         | Hybrid Vehicles     | 90                      | 2000                      |
| 5         | Trucks              | 50                      | 1500                      |
| 6         | Sports Cars         | 30                      | 3000                      |
| 7         | Compact Cars        | 110                     | 1000                      |
| 8         | Luxury Sedans       | 40                      | 3500                      |
| 9         | Vans                | 60                      | 1600                      |
| 10        | Pickup Trucks       | 35                      | 1700                      |

Total inventory capacity per day: $C_{tot} = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

#### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$ for all $i \in \{1,\ldots,10\}$

#### Objective

$$
\max \sum_{i=1}^{10} b_i x_i
$$

That is,
$$
\max \big(
1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
\big)
$$

#### Constraints

1. Per-type daily inventory limits:
   $$
   x_i \leq u_i \qquad \forall i \in \{1,\ldots,10\}
   $$
   That is,
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

2. Total inventory capacity:
   $$
   \sum_{i=1}^{10} x_i \leq 715
   $$

3. Nonnegativity and integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\}
   $$

#### Complete Model

$$
\begin{align*}
\max\quad & 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10} \\
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
& x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\}
\end{align*}
$$

Where the mapping of $i$ to VehicleType is as in the table above.