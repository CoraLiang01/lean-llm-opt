Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For each shelf $i$ (resource_id from capacity.csv), the capacity is $C_i$ (resource_capacity).
- For each product $j$ (item_name from products.csv), the value per unit is $v_j$ (item_value), and the weight per unit is $w_j$ (resource_requirement).

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

**Mathematical Model:**

Maximize total value:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{1,\ldots,20\}} v_j \cdot x_{ij}
$$

Subject to shelf capacity constraints (for each shelf $i$):
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in \{1,\ldots,10\}
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Where:**

- $v_j$ and $w_j$ are as given in the products.csv table above for each item_name $j$.
- $C_i$ is as given in the capacity.csv table above for each resource_id $i$.

**Decision variables:**

- $x_{ij}$: Number of units of product $j$ to place on shelf $i$, integer and $\geq 0$.