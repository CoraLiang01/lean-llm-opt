##### Objective Function:

$\quad \max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij}$

where $x_{ij}$ is the number of units of product $j$ placed on shelf $i$, and $v_j$ is the value of product $j$.

##### Constraints:

###### 1. Shelf Capacity Constraints:

$\sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i \in \{1,2,\ldots,10\}$

where $w_j$ is the weight of product $j$, and $C_i$ is the capacity of shelf $i$.

###### 2. Non-negativity and Integrality:

$x_{ij} \geq 0$ and integer, $\quad \forall i \in \{1,\ldots,10\}, \forall j \in \{1,\ldots,20\}$

---

##### Retrieved Information

**Shelf Capacities:**

| ShelfID | Capacity |
|---------|----------|
| 1       | 750      |
| 2       | 820      |
| 3       | 570      |
| 4       | 800      |
| 5       | 550      |
| 6       | 900      |
| 7       | 650      |
| 8       | 800      |
| 9       | 850      |
| 10      | 900      |

Let $C_i$ denote the capacity of shelf $i$ as above.

**Product Values and Weights:**

| ProductName | Value ($v_j$) | Weight ($w_j$) |
|-------------|--------------|---------------|
| 1           | 55           | 10            |
| 2           | 75           | 20            |
| 3           | 65           | 5             |
| 4           | 60           | 15            |
| 5           | 80           | 25            |
| 6           | 90           | 35            |
| 7           | 40           | 45            |
| 8           | 100          | 55            |
| 9           | 55           | 65            |
| 10          | 75           | 20            |
| 11          | 110          | 18            |
| 12          | 50           | 28            |
| 13          | 60           | 8             |
| 14          | 120          | 28            |
| 15          | 70           | 25            |
| 16          | 110          | 40            |
| 17          | 50           | 55            |
| 18          | 60           | 70            |
| 19          | 120          | 85            |
| 20          | 100          | 100           |

Let $v_j$ and $w_j$ denote the value and weight of product $j$ as above.

**Decision Variables:**

$x_{ij}$: Number of units of product $j$ placed on shelf $i$, for $i=1,\ldots,10$ and $j=1,\ldots,20$.

---

**Full Model:**

$\boxed{
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \, x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j \, x_{ij} \leq C_i \quad \forall i=1,\ldots,10 \\
& x_{ij} \geq 0,\ x_{ij} \in \mathbb{Z} \quad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
}$

where all parameters are as listed above.