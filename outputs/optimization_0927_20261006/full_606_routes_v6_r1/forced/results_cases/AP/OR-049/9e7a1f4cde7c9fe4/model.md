##### Decision Variables

Let $x_{ij}$ denote the number of units of product $j$ placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all shelves $i$ and products $j$.

##### Objective Function

$\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

where $v_j$ is the value of product $j$.

##### Constraints

For each shelf $i$ (where $i = 1, \ldots, 10$):

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i$

where $w_j$ is the weight of product $j$, and $C_i$ is the capacity of shelf $i$.

For all $i, j$:

$x_{ij} \in \mathbb{Z}_{\geq 0}$

---

##### Retrieved Information

**Shelf Capacities:**

| ShelfID | Capacity |
|---------|----------|
| 1       | 5.0      |
| 2       | 7.0      |
| 3       | 6.0      |
| 4       | 8.0      |
| 5       | 5.5      |
| 6       | 9.0      |
| 7       | 6.5      |
| 8       | 7.5      |
| 9       | 8.2      |
| 10      | 5.7      |

Let $C_1 = 5.0$, $C_2 = 7.0$, $C_3 = 6.0$, $C_4 = 8.0$, $C_5 = 5.5$, $C_6 = 9.0$, $C_7 = 6.5$, $C_8 = 7.5$, $C_9 = 8.2$, $C_{10} = 5.7$.

**Products:**

| Index ($j$) | Product Name           | Value ($v_j$) | Weight ($w_j$) |
|-------------|-----------------------|---------------|----------------|
| 1           | Smartphone            | 200           | 1.0            |
| 2           | Laptop                | 1500          | 5.0            |
| 3           | Headphones            | 100           | 0.5            |
| 4           | Camera                | 800           | 2.0            |
| 5           | Smartwatch            | 250           | 0.3            |
| 6           | Tablet                | 600           | 1.5            |
| 7           | Bluetooth Speaker     | 150           | 1.0            |
| 8           | Keyboard              | 80            | 0.8            |
| 9           | Mouse                 | 50            | 0.2            |
| 10          | Monitor               | 300           | 3.0            |
| 11          | Printer               | 400           | 4.0            |
| 12          | External Hard Drive   | 120           | 0.5            |
| 13          | Router                | 60            | 0.3            |
| 14          | Power Bank            | 40            | 0.4            |
| 15          | Memory Card           | 30            | 0.05           |
| 16          | USB Flash Drive       | 25            | 0.02           |
| 17          | Smart Home Hub        | 100           | 0.6            |
| 18          | Gaming Console        | 500           | 4.0            |
| 19          | Fitness Tracker       | 90            | 0.2            |
| 20          | E-Reader              | 180           | 0.5            |

So, for $j = 1, \ldots, 20$:

- $v_1 = 200$, $w_1 = 1.0$
- $v_2 = 1500$, $w_2 = 5.0$
- $v_3 = 100$, $w_3 = 0.5$
- $v_4 = 800$, $w_4 = 2.0$
- $v_5 = 250$, $w_5 = 0.3$
- $v_6 = 600$, $w_6 = 1.5$
- $v_7 = 150$, $w_7 = 1.0$
- $v_8 = 80$, $w_8 = 0.8$
- $v_9 = 50$, $w_9 = 0.2$
- $v_{10} = 300$, $w_{10} = 3.0$
- $v_{11} = 400$, $w_{11} = 4.0$
- $v_{12} = 120$, $w_{12} = 0.5$
- $v_{13} = 60$, $w_{13} = 0.3$
- $v_{14} = 40$, $w_{14} = 0.4$
- $v_{15} = 30$, $w_{15} = 0.05$
- $v_{16} = 25$, $w_{16} = 0.02$
- $v_{17} = 100$, $w_{17} = 0.6$
- $v_{18} = 500$, $w_{18} = 4.0$
- $v_{19} = 90$, $w_{19} = 0.2$
- $v_{20} = 180$, $w_{20} = 0.5$

##### Complete Model

$\boxed{
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i = 1, \ldots, 10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1, \ldots, 10,\; j = 1, \ldots, 20
\end{align*}
}$

where all $v_j$, $w_j$, and $C_i$ are as listed above.