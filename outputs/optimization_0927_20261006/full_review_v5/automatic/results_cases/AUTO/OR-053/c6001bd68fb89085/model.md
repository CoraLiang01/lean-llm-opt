Let $x_{ij}$ be the number of units of product $j$ (ProductName $j$) to be placed on shelf $i$ (ShelfID $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $S$ = set of shelves (ShelfID): $\{1,2,3,4,5,6,7,8,9,10\}$
- $P$ = set of products (ProductName): $\{1,2,3,\ldots,20\}$
- $c_i$ = capacity of shelf $i$ (from capacity.csv)
- $v_j$ = value of product $j$ (from products.csv)
- $w_j$ = weight of product $j$ (from products.csv)

**Data:**

From capacity.csv (in source order):

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

From products.csv (in source order):

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

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S, \; j \in P
$$

**Objective:**

$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i \in S$ (ShelfID):

$$
\sum_{j \in P} w_j \cdot x_{ij} \leq c_i
$$

where $c_i$ is the Capacity for ShelfID $i$.

**Variable Domains:**

$$
x_{ij} \in \{0,1,2,\ldots\} \quad \forall i \in S, \; j \in P
$$

---

**All parameters and identifiers are as retrieved and used in the model above.**