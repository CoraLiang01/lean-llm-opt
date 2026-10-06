Let $x_{ij}$ be the number of units of product $j$ (item_name from products.csv) to be placed on shelf $i$ (resource_id from capacity.csv). All $x_{ij}$ are nonnegative integers.

Parameters (from products.csv):

- For each product $j$:
    - item_value: $v_j$
    - resource_requirement: $w_j$

Parameters (from capacity.csv):

- For each shelf $i$:
    - resource_capacity: $C_i$

Indices:

- $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (resource_id from capacity.csv)
- $j \in \{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20\}$ (item_name from products.csv)

Objective:
\[
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{1,\ldots,20\}} v_j x_{ij}
\]
where $v_j$ is the item_value for product $j$.

Subject to, for each shelf $i$ (resource_id):

\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]
where $w_j$ is the resource_requirement for product $j$, and $C_i$ is the resource_capacity for shelf $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

---

Numerical Data:

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

Complete Model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j x_{i j} \leq C_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
\]
where $v_j$ and $w_j$ are as given above, and $C_i$ as per the table.