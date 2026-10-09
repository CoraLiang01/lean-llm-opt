Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (shelves, identified by resource_id)
- $j \in \{1,2,3,\ldots,20\}$ (products, identified by item_name)

**Parameters:**
- $v_j$: value of one unit of product $j$ (item_value)
- $w_j$: weight of one unit of product $j$ (resource_requirement)
- $C_i$: capacity of shelf $i$ (resource_capacity)

**Data:**

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

| item_name ($j$) | item_value ($v_j$) | resource_requirement ($w_j$) |
|-----------------|--------------------|------------------------------|
| 1               | 50                 | 10                           |
| 2               | 70                 | 20                           |
| 3               | 30                 | 5                            |
| 4               | 60                 | 15                           |
| 5               | 80                 | 25                           |
| 6               | 90                 | 30                           |
| 7               | 40                 | 12                           |
| 8               | 100                | 35                           |
| 9               | 55                 | 10                           |
| 10              | 75                 | 20                           |
| 11              | 65                 | 18                           |
| 12              | 95                 | 28                           |
| 13              | 45                 | 8                            |
| 14              | 85                 | 22                           |
| 15              | 70                 | 25                           |
| 16              | 110                | 40                           |
| 17              | 50                 | 14                           |
| 18              | 60                 | 16                           |
| 19              | 120                | 50                           |
| 20              | 100                | 30                           |

---

### Mathematical Model

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

**Subject to:**

- **Shelf capacity constraints:** For each shelf $i$,
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]

- **Integrality and nonnegativity:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,20\}
\]

---

**Where:**

- $x_{ij}$: Number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)
- $v_j$: Value of product $j$ (see table above)
- $w_j$: Weight of product $j$ (see table above)
- $C_i$: Capacity of shelf $i$ (see table above)

**All data and identifiers are as retrieved and preserved in original order.**