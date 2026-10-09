Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (from capacity.csv, in order):

  | resource_id |
  |-------------|
  | 1           |
  | 2           |
  | 3           |
  | 4           |
  | 5           |
  | 6           |
  | 7           |
  | 8           |
  | 9           |
  | 10          |

  Shelf capacities:

  $$
  \begin{align*}
  \text{resource\_capacity}_1 &= 500 \\
  \text{resource\_capacity}_2 &= 700 \\
  \text{resource\_capacity}_3 &= 600 \\
  \text{resource\_capacity}_4 &= 800 \\
  \text{resource\_capacity}_5 &= 550 \\
  \text{resource\_capacity}_6 &= 900 \\
  \text{resource\_capacity}_7 &= 650 \\
  \text{resource\_capacity}_8 &= 750 \\
  \text{resource\_capacity}_9 &= 820 \\
  \text{resource\_capacity}_{10} &= 570 \\
  \end{align*}
  $$

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
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

where $x_{ij}$ is the number of units of product $j$ (item_name $j$) to be placed on shelf $i$ (resource_id $i$).

---

**Objective:**

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{item\_value}_j \cdot x_{ij}
$$

where $\text{item\_value}_j$ is as given above for each $j$.

---

**Constraints:**

For each shelf $i$ (resource_id $i$):

$$
\sum_{j=1}^{20} \text{resource\_requirement}_j \cdot x_{ij} \leq \text{resource\_capacity}_i \qquad \forall i \in \{1,\ldots,10\}
$$

where $\text{resource\_requirement}_j$ and $\text{resource\_capacity}_i$ are as given above.

---

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

---

**Summary of Parameters (in source order):**

- Shelves (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Shelf capacities (resource_capacity): 500, 700, 600, 800, 550, 900, 650, 750, 820, 570
- Products (item_name): 1, 2, ..., 20
- Product values (item_value): 50, 70, 30, 60, 80, 90, 40, 100, 55, 75, 65, 95, 45, 85, 70, 110, 50, 60, 120, 100
- Product weights (resource_requirement): 10, 20, 5, 15, 25, 30, 12, 35, 10, 20, 18, 28, 8, 22, 25, 40, 14, 16, 50, 30

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} \text{item\_value}_j \cdot x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} \text{resource\_requirement}_j \cdot x_{ij} \leq \text{resource\_capacity}_i \quad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10,\ j = 1,\ldots,20
\end{align*}
$$

with all coefficients and identifiers as listed above.