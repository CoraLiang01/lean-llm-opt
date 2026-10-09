Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID from capacity.csv)
- $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of shelf $i$ (from capacity.csv)
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight of product $j$ (from products.csv)

**Data:**

Shelf capacities (from capacity.csv, in source order):

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

Product values and weights (from products.csv, in source order):

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
- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in \{1,\ldots,10\}$, $j \in \{1,\ldots,20\}$

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the value of product $j$ as given above.

**Constraints:**

For each shelf $i$:
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
\]
where $w_j$ is the weight of product $j$ and $c_i$ is the capacity of shelf $i$ as given above.

**Variable Domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

---

**All coefficients and identifiers are as retrieved and preserved in source order.**