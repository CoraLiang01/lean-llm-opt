Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let $S$ be the set of shelves (indexed by ShelfID), and $P$ be the set of products (indexed by ProductName).

Let $v_j$ be the value of product $j$, and $w_j$ be the weight of product $j$.

Let $C_i$ be the capacity of shelf $i$.

---

**Sets:**

- $S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}$
- $P = \{1, 2, 3, \ldots, 20\}$

**Parameters:**

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

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i \in S$:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i
$$

For all $i \in S$, $j \in P$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**Where:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Weight of product $j$ (see table above)
- $C_i$: Capacity of shelf $i$ (see table above)

---

**All data used:**

- Shelf capacities and IDs from capacity.csv (in source order)
- Product names, values, and weights from products.csv (in source order)
- Decision variables $x_{ij}$ are nonnegative integers as required by the problem statement.