**Mathematical Optimization Model**

Let $x_i$ be the number of vehicles of type $i$ to order per day, where $i$ indexes the 25 vehicle types listed in the table below.

**Parameters:**

- $v_i$ = profit from selling one unit of vehicle type $i$ (from "Value" column)
- $w_i$ = inventory space required for one unit of vehicle type $i$ (from "Weight" column)
- $C$ = total inventory capacity (from "Capacity" column)

**Data:**

| $i$ | ProductName           | $v_i$ (Value) | $w_i$ (Weight) |
|----|-----------------------|--------------|---------------|
| 1  | Sedan                 | 2524         | 99            |
| 2  | SUV                   | 4614         | 55            |
| 3  | Truck                 | 8416         | 75            |
| 4  | Convertible           | 5917         | 94            |
| 5  | Minivan               | 9048         | 80            |
| 6  | Coupe                 | 1140         | 82            |
| 7  | Hatchback             | 8962         | 71            |
| 8  | Station Wagon         | 1888         | 100           |
| 9  | Electric Car          | 8487         | 28            |
| 10 | Hybrid Car            | 4425         | 93            |
| 11 | Luxury Sedan          | 4717         | 84            |
| 12 | Sports Car            | 4210         | 83            |
| 13 | Crossover             | 1226         | 62            |
| 14 | Diesel Truck          | 7400         | 90            |
| 15 | Compact SUV           | 4639         | 99            |
| 16 | Luxury SUV            | 7712         | 96            |
| 17 | Cargo Van             | 3299         | 21            |
| 18 | Pickup Truck          | 9895         | 39            |
| 19 | Roadster              | 4496         | 99            |
| 20 | Muscle Car            | 4526         | 81            |
| 21 | Off-road Vehicle      | 5688         | 6             |
| 22 | Camper Van            | 3007         | 58            |
| 23 | Compact Car           | 3623         | 37            |
| 24 | Motorcycle            | 8474         | 15            |
| 25 | Electric SUV          | 8372         | 37            |

Total inventory capacity: $C = 765$

---

**Decision Variables:**

- $x_i \in \mathbb{Z}_{\geq 0}$, for $i = 1, \ldots, 25$

---

**Objective:**

\[
\max \sum_{i=1}^{25} v_i x_i
\]

That is,

\[
\max \Big(
2524\,x_1 + 4614\,x_2 + 8416\,x_3 + 5917\,x_4 + 9048\,x_5 + 1140\,x_6 + 8962\,x_7 + 1888\,x_8 + 8487\,x_9 + 4425\,x_{10} + 4717\,x_{11} + 4210\,x_{12} + 1226\,x_{13} + 7400\,x_{14} + 4639\,x_{15} + 7712\,x_{16} + 3299\,x_{17} + 9895\,x_{18} + 4496\,x_{19} + 4526\,x_{20} + 5688\,x_{21} + 3007\,x_{22} + 3623\,x_{23} + 8474\,x_{24} + 8372\,x_{25}
\Big)
\]

---

**Subject to:**

**Inventory Capacity Constraint:**

\[
\sum_{i=1}^{25} w_i x_i \leq 765
\]

That is,

\[
99\,x_1 + 55\,x_2 + 75\,x_3 + 94\,x_4 + 80\,x_5 + 82\,x_6 + 71\,x_7 + 100\,x_8 + 28\,x_9 + 93\,x_{10} + 84\,x_{11} + 83\,x_{12} + 62\,x_{13} + 90\,x_{14} + 99\,x_{15} + 96\,x_{16} + 21\,x_{17} + 39\,x_{18} + 99\,x_{19} + 81\,x_{20} + 6\,x_{21} + 58\,x_{22} + 37\,x_{23} + 15\,x_{24} + 37\,x_{25} \leq 765
\]

**Integrality and Nonnegativity:**

\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1, \ldots, 25
\]

---

**Complete Model:**

\[
\begin{align*}
\max\quad & 2524\,x_1 + 4614\,x_2 + 8416\,x_3 + 5917\,x_4 + 9048\,x_5 + 1140\,x_6 + 8962\,x_7 + 1888\,x_8 \\
& + 8487\,x_9 + 4425\,x_{10} + 4717\,x_{11} + 4210\,x_{12} + 1226\,x_{13} + 7400\,x_{14} + 4639\,x_{15} \\
& + 7712\,x_{16} + 3299\,x_{17} + 9895\,x_{18} + 4496\,x_{19} + 4526\,x_{20} + 5688\,x_{21} + 3007\,x_{22} \\
& + 3623\,x_{23} + 8474\,x_{24} + 8372\,x_{25} \\
\text{s.t.}\quad & 99\,x_1 + 55\,x_2 + 75\,x_3 + 94\,x_4 + 80\,x_5 + 82\,x_6 + 71\,x_7 + 100\,x_8 \\
& + 28\,x_9 + 93\,x_{10} + 84\,x_{11} + 83\,x_{12} + 62\,x_{13} + 90\,x_{14} + 99\,x_{15} \\
& + 96\,x_{16} + 21\,x_{17} + 39\,x_{18} + 99\,x_{19} + 81\,x_{20} + 6\,x_{21} + 58\,x_{22} \\
& + 37\,x_{23} + 15\,x_{24} + 37\,x_{25} \leq 765 \\
& x_i \in \mathbb{Z}_{\geq 0}, \quad i = 1, \ldots, 25
\end{align*}
\]