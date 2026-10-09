Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$.

**Parameters:**

- Shelves (from capacity.csv):

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

- Products (from products.csv):

  | ProductName | Value | Weight |
  |-------------|-------|--------|
  | 1           | 55    | 10     |
  | 2           | 75    | 20     |
  | 3           | 65    | 5      |
  | 4           | 60    | 15     |
  | 5           | 80    | 25     |
  | 6           | 90    | 35     |
  | 7           | 40    | 45     |
  | 8           | 100   | 55     |
  | 9           | 55    | 65     |
  | 10          | 75    | 20     |
  | 11          | 110   | 18     |
  | 12          | 50    | 28     |
  | 13          | 60    | 8      |
  | 14          | 120   | 28     |
  | 15          | 70    | 25     |
  | 16          | 110   | 40     |
  | 17          | 50    | 55     |
  | 18          | 60    | 70     |
  | 19          | 120   | 85     |
  | 20          | 100   | 100    |

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$.

**Constraints:**

For each shelf $i$ (with capacity $C_i$):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the Weight of product $j$ and $C_i$ is the Capacity of shelf $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**Parameter Table (for reference):**

- $C_1 = 750$, $C_2 = 820$, $C_3 = 570$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 800$, $C_9 = 850$, $C_{10} = 900$
- $(v_j, w_j)$ for $j=1$ to $20$ as listed above.

---

**Summary:**

Maximize total value of products allocated to shelves, subject to each shelf's capacity, with integer nonnegative allocation variables for each product-shelf pair. All coefficients and identifiers are as retrieved.