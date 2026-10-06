Let $x_{ij}$ be the number of units of product $j$ (item_name from products.csv) to be placed on shelf $i$ (resource_id from capacity.csv). All $x_{ij}$ are integer and $\geq 0$.

Maximize total value:
$$
\max \sum_{i \in S} \sum_{j \in P} v_j \cdot x_{ij}
$$

where:
- $S$ is the set of shelves (resource_id from capacity.csv)
- $P$ is the set of products (item_name from products.csv)
- $v_j$ is the value of product $j$ (item_value from products.csv)

Subject to shelf capacity constraints:
$$
\sum_{j \in P} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in S
$$

where:
- $w_j$ is the resource requirement (weight) of product $j$ (resource_requirement from products.csv)
- $C_i$ is the capacity of shelf $i$ (resource_capacity from capacity.csv)

And integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
$$

---

Numerical Data:

**Shelves (from capacity.csv):**

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

**Products (from products.csv):**

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

**Complete Model:**

Maximize
$$
\sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

Subject to, for each shelf $i$:
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,20
$$

where $v_j$ and $w_j$ are as given above, and $C_i$ is the resource_capacity for shelf $i$.