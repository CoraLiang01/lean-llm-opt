Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (resource_id from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (item_name from products.csv).

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the item_value of product $j$ from products.csv.

Subject to, for each shelf $i$ (resource_id):

Capacity constraints:
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,\ldots,10\}
\]
where $w_j$ is the resource_requirement of product $j$ from products.csv, and $c_i$ is the resource_capacity of shelf $i$ from capacity.csv.

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\; j \in \{1,2,\ldots,20\}
\]

---

#### Data

**Shelves (from capacity.csv):**

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

**Products (from products.csv):**

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
& \sum_{j=1}^{20} w_j x_{ij} \leq c_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,10;\ j=1,\ldots,20
\end{align*}
\]
where for each $j$:
- $v_j$ is item_value from products.csv,
- $w_j$ is resource_requirement from products.csv,
and for each $i$:
- $c_i$ is resource_capacity from capacity.csv,
with all indices and coefficients as shown in the tables above.