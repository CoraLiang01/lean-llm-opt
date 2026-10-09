Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the VehicleID in the order given below.

**Objective:**

$$
\max \; 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
$$

**Subject to:**

_Per-vehicle-type daily inventory limits:_
\[
\begin{align*}
x_1 &\leq 100 \quad &\text{(Sedans)} \\
x_2 &\leq 80 \quad &\text{(SUVs)} \\
x_3 &\leq 120 \quad &\text{(Electric Vehicles)} \\
x_4 &\leq 90 \quad &\text{(Hybrid Vehicles)} \\
x_5 &\leq 50 \quad &\text{(Trucks)} \\
x_6 &\leq 30 \quad &\text{(Sports Cars)} \\
x_7 &\leq 110 \quad &\text{(Compact Cars)} \\
x_8 &\leq 40 \quad &\text{(Luxury Sedans)} \\
x_9 &\leq 60 \quad &\text{(Vans)} \\
x_{10} &\leq 35 \quad &\text{(Pickup Trucks)} \\
\end{align*}
\]

_Total inventory capacity per day:_
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq  \text{(total inventory capacity)}
\]
(Note: The total inventory capacity value must be specified by the user or business context; if not provided, this constraint should be included with the appropriate right-hand side when known.)

_Nonnegativity and integrality:_
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
\]

---

**Parameter Table (source order):**

| VehicleID | VehicleType        | Per-Type Capacity | Benefit Coefficient |
|-----------|-------------------|-------------------|---------------------|
| 1         | Sedans            | 100               | 1200                |
| 2         | SUVs              | 80                | 1800                |
| 3         | Electric Vehicles | 120               | 2500                |
| 4         | Hybrid Vehicles   | 90                | 2000                |
| 5         | Trucks            | 50                | 1500                |
| 6         | Sports Cars       | 30                | 3000                |
| 7         | Compact Cars      | 110               | 1000                |
| 8         | Luxury Sedans     | 40                | 3500                |
| 9         | Vans              | 60                | 1600                |
| 10        | Pickup Trucks     | 35                | 1700                |

**Decision variables:**  
$x_i$ = number of vehicles of type $i$ to order per day (integer, $\geq 0$).