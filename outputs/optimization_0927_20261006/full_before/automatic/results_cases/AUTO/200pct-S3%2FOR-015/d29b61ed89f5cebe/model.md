Let $x_{ij}$ be the number of units of product $j$ (item_name $j$) to be placed on shelf $i$ (resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- For each shelf $i$ (resource_id): resource_capacity $c_i$
- For each product $j$ (item_name): item_value $v_j$, resource_requirement $a_j$

**Objective:**
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the item_value for product $j$.

**Constraints:**

For each shelf $i$ (resource_id $i$):
\[
\sum_{j=1}^{20} a_j \cdot x_{ij} \leq c_i
\]
where $a_j$ is the resource_requirement for product $j$, and $c_i$ is the resource_capacity for shelf $i$.

For all $i=1,\ldots,10$ and $j=1,\ldots,20$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

---

**Data Used (in source order):**

*Shelves (capacity.csv):*

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

*Products (products.csv):*

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

**Complete Model:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} a_j x_{ij} \leq c_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
\]
where $v_j$ and $a_j$ are as listed above, and $c_i$ as per the shelf data.