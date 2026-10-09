Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- $V_j$: value of product $j$ (item_value)
- $W_j$: weight (resource requirement) of product $j$ (resource_requirement)
- $C_i$: capacity of shelf $i$ (resource_capacity)

**Sets:**

- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (resource_id from capacity.csv)
- $j \in \{1,2,3,\ldots,20\}$ (item_name from products.csv)

---

### Objective

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} V_j \cdot x_{ij}
$$

where $V_j$ is as follows:

| item_name | item_value ($V_j$) |
|-----------|-------------------|
| 1         | 50                |
| 2         | 70                |
| 3         | 30                |
| 4         | 60                |
| 5         | 80                |
| 6         | 90                |
| 7         | 40                |
| 8         | 100               |
| 9         | 55                |
| 10        | 75                |
| 11        | 65                |
| 12        | 95                |
| 13        | 45                |
| 14        | 85                |
| 15        | 70                |
| 16        | 110               |
| 17        | 50                |
| 18        | 60                |
| 19        | 120               |
| 20        | 100               |

---

### Constraints

#### Shelf Capacity Constraints

For each shelf $i$ (resource_id):

$$
\sum_{j=1}^{20} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
$$

where $W_j$ is as follows:

| item_name | resource_requirement ($W_j$) |
|-----------|------------------------------|
| 1         | 10                           |
| 2         | 20                           |
| 3         | 5                            |
| 4         | 15                           |
| 5         | 25                           |
| 6         | 30                           |
| 7         | 12                           |
| 8         | 35                           |
| 9         | 10                           |
| 10        | 20                           |
| 11        | 18                           |
| 12        | 28                           |
| 13        | 8                            |
| 14        | 22                           |
| 15        | 25                           |
| 16        | 40                           |
| 17        | 14                           |
| 18        | 16                           |
| 19        | 50                           |
| 20        | 30                           |

and $C_i$ is as follows:

| resource_id ($i$) | resource_capacity ($C_i$) |
|-------------------|--------------------------|
| 1                 | 500                      |
| 2                 | 700                      |
| 3                 | 600                      |
| 4                 | 800                      |
| 5                 | 550                      |
| 6                 | 900                      |
| 7                 | 650                      |
| 8                 | 750                      |
| 9                 | 820                      |
| 10                | 570                      |

#### Integrality and Nonnegativity

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,20\}
$$

---

**Summary of Model:**

- Decision variables: $x_{ij}$ = number of units of product $j$ on shelf $i$, integer, $\geq 0$
- Objective: maximize total value across all shelves and products
- For each shelf, total weight of products cannot exceed its capacity

All coefficients and identifiers are as retrieved and preserved from the original data.