Let $x_{ij}$ be the number of units of product $j$ (ProductName $j$) to be placed on shelf $i$ (ShelfID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv):

  | ShelfID | Capacity |
  |---------|----------|
  | 1       | 500      |
  | 2       | 700      |
  | 3       | 600      |
  | 4       | 800      |
  | 5       | 550      |
  | 6       | 900      |
  | 7       | 650      |
  | 8       | 750      |
  | 9       | 820      |
  | 10      | 570      |

- Products (from products.csv):

  | ProductName | Value | Weight |
  |-------------|-------|--------|
  | 1           | 50    | 10     |
  | 2           | 70    | 20     |
  | 3           | 30    | 5      |
  | 4           | 60    | 15     |
  | 5           | 80    | 25     |
  | 6           | 90    | 30     |
  | 7           | 40    | 12     |
  | 8           | 100   | 35     |
  | 9           | 55    | 10     |
  | 10          | 75    | 20     |
  | 11          | 65    | 18     |
  | 12          | 95    | 28     |
  | 13          | 45    | 8      |
  | 14          | 85    | 22     |
  | 15          | 70    | 25     |
  | 16          | 110   | 40     |
  | 17          | 50    | 14     |
  | 18          | 60    | 16     |
  | 19          | 120   | 50     |
  | 20          | 100   | 30     |

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of ProductName $j$.

**Constraints:**

For each shelf $i$ (ShelfID $i$):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the Weight of ProductName $j$, and $C_i$ is the Capacity of ShelfID $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**Parameter Values:**

- $C_1 = 500$, $C_2 = 700$, $C_3 = 600$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 750$, $C_9 = 820$, $C_{10} = 570$
- $(v_j, w_j)$ for $j=1$ to $20$ as listed above.

---

**Summary Table of Variables and Parameters:**

| $i$ (ShelfID) | $C_i$ |
|---------------|-------|
| 1             | 500   |
| 2             | 700   |
| 3             | 600   |
| 4             | 800   |
| 5             | 550   |
| 6             | 900   |
| 7             | 650   |
| 8             | 750   |
| 9             | 820   |
| 10            | 570   |

| $j$ (ProductName) | $v_j$ (Value) | $w_j$ (Weight) |
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