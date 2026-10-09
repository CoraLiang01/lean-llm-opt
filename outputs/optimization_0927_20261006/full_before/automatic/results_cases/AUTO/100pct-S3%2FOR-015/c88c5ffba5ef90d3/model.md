Let $x_{ij}$ be the number of units of product $j$ (item_name $j$) to be placed on shelf $i$ (resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Shelf capacities (resource_capacity):

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

- Products (item_name): 1, 2, ..., 20
- Product values (item_value) and weights (resource_requirement):

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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{item\_value}_j \cdot x_{ij}
$$

where $\text{item\_value}_j$ is as given above for each $j$.

**Constraints:**

For each shelf $i$ (resource_id $i$):

$$
\sum_{j=1}^{20} \text{resource\_requirement}_j \cdot x_{ij} \leq \text{resource\_capacity}_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $\text{resource\_requirement}_j$ and $\text{resource\_capacity}_i$ are as given above.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**All coefficients and identifiers are as retrieved and shown above.**