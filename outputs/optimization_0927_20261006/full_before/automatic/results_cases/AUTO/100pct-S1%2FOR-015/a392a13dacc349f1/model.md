Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For each shelf $i$ (resource_id from capacity.csv), let $C_i$ be its resource_capacity.
- For each product $j$ (item_name from products.csv), let $v_j$ be its item_value and $w_j$ its resource_requirement.

**Data:**

From capacity.csv (in source order):

| resource_id | resource_capacity |
|-------------|------------------|
| 1           | 500              |
| 2           | 700              |
| 3           | 600              |
| 4           | 800              |
| 5           | 550              |
| 6           | 900              |
| 7           | 650              |
| 8           | 750              |
| 9           | 820              |
| 10          | 570              |

From products.csv (in source order):

| item_name | item_value | resource_requirement |
|-----------|------------|---------------------|
| 1         | 50         | 10                  |
| 2         | 70         | 20                  |
| 3         | 30         | 5                   |
| 4         | 60         | 15                  |
| 5         | 80         | 25                  |
| 6         | 90         | 30                  |
| 7         | 40         | 12                  |
| 8         | 100        | 35                  |
| 9         | 55         | 10                  |
| 10        | 75         | 20                  |
| 11        | 65         | 18                  |
| 12        | 95         | 28                  |
| 13        | 45         | 8                   |
| 14        | 85         | 22                  |
| 15        | 70         | 25                  |
| 16        | 110        | 40                  |
| 17        | 50         | 14                  |
| 18        | 60         | 16                  |
| 19        | 120        | 50                  |
| 20        | 100        | 30                  |

---

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall\, i \in \{1,2,\ldots,10\},\ j \in \{1,2,\ldots,20\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j\, x_{ij}
$$
where $v_j$ is the item_value of product $j$.

**Constraints:**

For each shelf $i$ (resource_id from 1 to 10):
$$
\sum_{j=1}^{20} w_j\, x_{ij} \leq C_i
$$
where $w_j$ is the resource_requirement of product $j$, and $C_i$ is the resource_capacity of shelf $i$.

**Explicitly, for each shelf:**

- Shelf 1: $\sum_{j=1}^{20} w_j\, x_{1j} \leq 500$
- Shelf 2: $\sum_{j=1}^{20} w_j\, x_{2j} \leq 700$
- Shelf 3: $\sum_{j=1}^{20} w_j\, x_{3j} \leq 600$
- Shelf 4: $\sum_{j=1}^{20} w_j\, x_{4j} \leq 800$
- Shelf 5: $\sum_{j=1}^{20} w_j\, x_{5j} \leq 550$
- Shelf 6: $\sum_{j=1}^{20} w_j\, x_{6j} \leq 900$
- Shelf 7: $\sum_{j=1}^{20} w_j\, x_{7j} \leq 650$
- Shelf 8: $\sum_{j=1}^{20} w_j\, x_{8j} \leq 750$
- Shelf 9: $\sum_{j=1}^{20} w_j\, x_{9j} \leq 820$
- Shelf 10: $\sum_{j=1}^{20} w_j\, x_{10j} \leq 570$

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall\, i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Where:**

- $v_j$ and $w_j$ are as given in products.csv for item_name $j$.
- $C_i$ is as given in capacity.csv for resource_id $i$.

This model maximizes the total value of products allocated to shelves, subject to each shelf's weight capacity, with integer allocation decisions for each product-shelf pair.