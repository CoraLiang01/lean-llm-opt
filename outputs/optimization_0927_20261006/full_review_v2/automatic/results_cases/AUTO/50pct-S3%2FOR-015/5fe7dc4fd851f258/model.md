Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$. All $x_{ij}$ are required to be nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv, in order):

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

- Products (from products.csv, in order):

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

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective Function:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value of product $j$.

**Constraints:**

For each shelf $i$ (resource_id $i$):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the resource_requirement of product $j$, and $c_i$ is the resource_capacity of shelf $i$.

**Explicitly, using the data:**

For each $i = 1,\ldots,10$ (shelf/resource_id):

$$
\sum_{j=1}^{20} \text{resource\_requirement}_j \cdot x_{ij} \leq \text{resource\_capacity}_i
$$

with:

- resource_capacity = [500, 700, 600, 800, 550, 900, 650, 750, 820, 570] for $i=1$ to $10$
- resource_requirement = [10, 20, 5, 15, 25, 30, 12, 35, 10, 20, 18, 28, 8, 22, 25, 40, 14, 16, 50, 30] for $j=1$ to $20$
- item_value = [50, 70, 30, 60, 80, 90, 40, 100, 55, 75, 65, 95, 45, 85, 70, 110, 50, 60, 120, 100] for $j=1$ to $20$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,20
$$

**Summary Table of Parameters:**

| Shelf (resource_id) | Capacity (resource_capacity) |
|---------------------|-----------------------------|
| 1                   | 500                         |
| 2                   | 700                         |
| 3                   | 600                         |
| 4                   | 800                         |
| 5                   | 550                         |
| 6                   | 900                         |
| 7                   | 650                         |
| 8                   | 750                         |
| 9                   | 820                         |
| 10                  | 570                         |

| Product (item_name) | Value (item_value) | Weight (resource_requirement) |
|---------------------|-------------------|------------------------------|
| 1                   | 50                | 10                           |
| 2                   | 70                | 20                           |
| 3                   | 30                | 5                            |
| 4                   | 60                | 15                           |
| 5                   | 80                | 25                           |
| 6                   | 90                | 30                           |
| 7                   | 40                | 12                           |
| 8                   | 100               | 35                           |
| 9                   | 55                | 10                           |
| 10                  | 75                | 20                           |
| 11                  | 65                | 18                           |
| 12                  | 95                | 28                           |
| 13                  | 45                | 8                            |
| 14                  | 85                | 22                           |
| 15                  | 70                | 25                           |
| 16                  | 110               | 40                           |
| 17                  | 50                | 14                           |
| 18                  | 60                | 16                           |
| 19                  | 120               | 50                           |
| 20                  | 100               | 30                           |