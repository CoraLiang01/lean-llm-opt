Let $x_i$ be the number of vehicles of type $i$ to order per day. Each $x_i$ is a nonnegative integer.

Let $I$ be the set of vehicle types, indexed by VehicleID as in the data.

Let $b_i$ be the benefit coefficient for vehicle type $i$ (from "products.csv", Value column, matched by VehicleType/ProductName).

Let $u_i$ be the daily inventory limit for vehicle type $i$ (from "capacity.csv", Capacity column, matched by VehicleID).

Let $C = \sum_{i \in I} u_i$ be the total inventory capacity per day (sum of all per-type capacities).

#### Sets and Parameters

| VehicleID | VehicleType        | $b_i$ (Value) | $u_i$ (Capacity) |
|-----------|--------------------|---------------|------------------|
| 1         | Sedans             | 1200          | 100              |
| 2         | SUVs               | 1800          | 80               |
| 3         | Electric Vehicles  | 2500          | 120              |
| 4         | Hybrid Vehicles    | 2000          | 90               |
| 5         | Trucks             | 1500          | 50               |
| 6         | Sports Cars        | 3000          | 30               |
| 7         | Compact Cars       | 1000          | 110              |
| 8         | Luxury Sedans      | 3500          | 40               |
| 9         | Vans               | 1600          | 60               |
| 10        | Pickup Trucks      | 1700          | 35               |

Total inventory capacity per day: $C = 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715$

#### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$ (VehicleID 1 to 10)

#### Objective

$$
\max \sum_{i \in I} b_i x_i
$$

That is,
$$
\max \big(
1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
\big)
$$

#### Constraints

1. **Per-vehicle-type daily inventory limits:**
   $$
   x_i \leq u_i, \quad \forall i \in I
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

2. **Total inventory capacity:**
   $$
   \sum_{i \in I} x_i \leq 715
   $$

3. **Nonnegativity and integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
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
& x_i \in \mathbb{Z}_{\geq 0}, \quad i = 1,\ldots,10
\end{align*}
$$

Where the mapping of $i$ to vehicle type is as in the table above.