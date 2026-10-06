Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes shelves by resource_id from capacity.csv and $j$ indexes products by item_name from products.csv. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv, in order):

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

- Products (from products.csv, in order):

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

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of product } j \text{ (item_name as above) placed on shelf } i \text{ (resource_id as above)}, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{1,\ldots,20\}} v_j \cdot x_{ij}
$$

where $v_j$ is the item_value of product $j$.

**Constraints:**

For each shelf $i$ (resource_id from 1 to 10):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $w_j$ is the resource_requirement of product $j$, and $C_i$ is the resource_capacity of shelf $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\; j \in \{1,\ldots,20\}
$$

---

**Parameter Table (for reference):**

- For $i$ (shelf/resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- For $j$ (product/item_name): 1, 2, ..., 20
- $v_j$ (item_value): 50, 70, 30, 60, 80, 90, 40, 100, 55, 75, 65, 95, 45, 85, 70, 110, 50, 60, 120, 100
- $w_j$ (resource_requirement): 10, 20, 5, 15, 25, 30, 12, 35, 10, 20, 18, 28, 8, 22, 25, 40, 14, 16, 50, 30
- $C_i$ (resource_capacity): 500, 700, 600, 800, 550, 900, 650, 750, 820, 570

---

**Complete Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10,\; j = 1,\ldots,20
\end{align*}
$$

where all coefficients and indices are as listed above, preserving the original file order and identifiers.