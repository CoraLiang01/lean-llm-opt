Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

**Parameters:**

- Shelves (from capacity.csv):

| resource_id | resource_capacity |
|-------------|------------------|
|      1      |       500        |
|      2      |       700        |
|      3      |       600        |
|      4      |       800        |
|      5      |       550        |
|      6      |       900        |
|      7      |       650        |
|      8      |       750        |
|      9      |       820        |
|     10      |       570        |

- Products (from products.csv):

| item_name | item_value | resource_requirement |
|-----------|------------|---------------------|
|     1     |     50     |         10          |
|     2     |     70     |         20          |
|     3     |     30     |         5           |
|     4     |     60     |         15          |
|     5     |     80     |         25          |
|     6     |     90     |         30          |
|     7     |     40     |         12          |
|     8     |    100     |         35          |
|     9     |     55     |         10          |
|    10     |     75     |         20          |
|    11     |     65     |         18          |
|    12     |     95     |         28          |
|    13     |     45     |         8           |
|    14     |     85     |         22          |
|    15     |     70     |         25          |
|    16     |    110     |         40          |
|    17     |     50     |         14          |
|    18     |     60     |         16          |
|    19     |    120     |         50          |
|    20     |    100     |         30          |

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
where $v_j$ is the item_value of product $j$.

**Constraints:**

For each shelf $i$ (resource_id $i$):
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10
$$
where $w_j$ is the resource_requirement of product $j$, and $c_i$ is the resource_capacity of shelf $i$.

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Explicit Data:**

- For $i=1$ (resource_id 1, resource_capacity 500):
  $$
  \sum_{j=1}^{20} w_j x_{1j} \leq 500
  $$
- For $i=2$ (resource_id 2, resource_capacity 700):
  $$
  \sum_{j=1}^{20} w_j x_{2j} \leq 700
  $$
- ...
- For $i=10$ (resource_id 10, resource_capacity 570):
  $$
  \sum_{j=1}^{20} w_j x_{10j} \leq 570
  $$

Where for each $j$ (item_name):

| $j$ | $v_j$ | $w_j$ |
|-----|-------|-------|
|  1  |  50   |  10   |
|  2  |  70   |  20   |
|  3  |  30   |   5   |
|  4  |  60   |  15   |
|  5  |  80   |  25   |
|  6  |  90   |  30   |
|  7  |  40   |  12   |
|  8  | 100   |  35   |
|  9  |  55   |  10   |
| 10  |  75   |  20   |
| 11  |  65   |  18   |
| 12  |  95   |  28   |
| 13  |  45   |   8   |
| 14  |  85   |  22   |
| 15  |  70   |  25   |
| 16  | 110   |  40   |
| 17  |  50   |  14   |
| 18  |  60   |  16   |
| 19  | 120   |  50   |
| 20  | 100   |  30   |

---

**Summary:**

Maximize total value of products allocated to shelves, subject to each shelf's capacity, using integer variables $x_{ij}$ for each shelf $i$ and product $j$, with the above coefficients and constraints.