##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$, for $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,20\}$.

##### Parameters

- Shelf capacities (from "capacity.csv"):
  - Shelf 1: $C_1 = 500$
  - Shelf 2: $C_2 = 700$
  - Shelf 3: $C_3 = 600$
  - Shelf 4: $C_4 = 800$
  - Shelf 5: $C_5 = 550$
  - Shelf 6: $C_6 = 900$
  - Shelf 7: $C_7 = 650$
  - Shelf 8: $C_8 = 750$
  - Shelf 9: $C_9 = 820$
  - Shelf 10: $C_{10} = 570$

- Product values and weights (from "products.csv"):

| Product $j$ | Value $v_j$ | Weight $w_j$ |
|:-----------:|:-----------:|:------------:|
| 1           | 50          | 10           |
| 2           | 70          | 20           |
| 3           | 30          | 5            |
| 4           | 60          | 15           |
| 5           | 80          | 25           |
| 6           | 90          | 30           |
| 7           | 40          | 12           |
| 8           | 100         | 35           |
| 9           | 55          | 10           |
| 10          | 75          | 20           |
| 11          | 65          | 18           |
| 12          | 95          | 28           |
| 13          | 45          | 8            |
| 14          | 85          | 22           |
| 15          | 70          | 25           |
| 16          | 110         | 40           |
| 17          | 50          | 14           |
| 18          | 60          | 16           |
| 19          | 120         | 50           |
| 20          | 100         | 30           |

##### Objective Function

$\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}$

##### Constraints

For each shelf $i = 1,2,\ldots,10$:
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i
$$

For all $i=1,\ldots,10$ and $j=1,\ldots,20$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

##### Complete Model

$\boxed{
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
}$

Where $C_i$, $v_j$, and $w_j$ are as listed above.