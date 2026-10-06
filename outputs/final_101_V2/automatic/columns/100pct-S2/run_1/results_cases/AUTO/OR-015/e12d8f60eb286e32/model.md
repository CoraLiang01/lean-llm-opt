Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

Indices:
- $i$ indexes shelves, with resource_id from capacity.csv: $i \in \{1,2,3,4,5,6,7,8,9,10\}$
- $j$ indexes products, with item_name from products.csv: $j \in \{1,2,\ldots,20\}$

Parameters:
- $v_j$: item_value of product $j$ (from products.csv)
- $w_j$: resource_requirement of product $j$ (from products.csv)
- $C_i$: resource_capacity of shelf $i$ (from capacity.csv)

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

Subject to (for each shelf $i$):
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

---

Numerical Data:

Shelves (from capacity.csv):

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

Products (from products.csv):

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
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

subject to, for each $i$:

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

where $v_j$ and $w_j$ are as listed above, and $C_i$ as per the table.