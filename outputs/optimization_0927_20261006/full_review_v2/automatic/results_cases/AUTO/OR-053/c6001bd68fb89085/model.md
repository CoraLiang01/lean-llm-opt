Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

**Parameters:**

- Shelves ($i$):  
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10  
  with capacities:  
  $C_1 = 500$, $C_2 = 700$, $C_3 = 600$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 750$, $C_9 = 820$, $C_{10} = 570$

- Products ($j$):  
  1, 2, ..., 20  
  with values $v_j$ and weights $w_j$ as follows:

| ProductName ($j$) | Value ($v_j$) | Weight ($w_j$) |
|-------------------|--------------|---------------|
| 1                 | 50           | 10            |
| 2                 | 70           | 20            |
| 3                 | 30           | 5             |
| 4                 | 60           | 15            |
| 5                 | 80           | 25            |
| 6                 | 90           | 30            |
| 7                 | 40           | 12            |
| 8                 | 100          | 35            |
| 9                 | 55           | 10            |
| 10                | 75           | 20            |
| 11                | 65           | 18            |
| 12                | 95           | 28            |
| 13                | 45           | 8             |
| 14                | 85           | 22            |
| 15                | 70           | 25            |
| 16                | 110          | 40            |
| 17                | 50           | 14            |
| 18                | 60           | 16            |
| 19                | 120          | 50            |
| 20                | 100          | 30            |

---

### Mathematical Model

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i = 1, \ldots, 10$:
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i
$$

For all $i = 1, \ldots, 10$, $j = 1, \ldots, 20$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Where:**

- $v_j$ and $w_j$ are as given in the table above.
- $C_i$ is the capacity of shelf $i$ as listed above.
- $x_{ij}$ is the integer number of units of product $j$ placed on shelf $i$.