Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For each shelf $i$ (resource_id from capacity.csv), the capacity is $C_i$ (resource_capacity).
- For each product $j$ (item_name from products.csv), the value per unit is $v_j$ (item_value), and the weight per unit is $w_j$ (resource_requirement).

**Sets:**

- $I$ = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} (resource_id from capacity.csv)
- $J$ = {1, 2, ..., 20} (item_name from products.csv)

**Objective:**

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

**Subject to:**

For each shelf $i$ (resource_id):

$$
\sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
$$

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

**Parameter Data (in source order):**

*From capacity.csv:*

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

*From products.csv:*

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

**Summary of Model:**

- Decision variables: $x_{ij}$ = number of units of product $j$ on shelf $i$, integer, $\geq 0$
- Objective: maximize total value across all shelves and products
- For each shelf, total weight of products cannot exceed its resource_capacity

This is a complete integer programming model for the BigMart Sales shelf allocation problem using the provided data.