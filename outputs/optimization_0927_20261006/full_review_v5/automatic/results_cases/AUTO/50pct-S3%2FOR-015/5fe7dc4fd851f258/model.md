#### Sets and Indices

- Let $I$ be the set of shelves, indexed by $i$ (corresponding to resource_id in capacity.csv).
- Let $J$ be the set of products, indexed by $j$ (corresponding to item_name in products.csv).

#### Parameters

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

Let:
- $c_i$ = resource_capacity of shelf $i$
- $v_j$ = item_value of product $j$
- $a_j$ = resource_requirement (weight) of product $j$

#### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ to place on shelf $i$

#### Objective Function

$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

#### Constraints

1. **Shelf Capacity Constraints** (for each shelf $i$):

   $$
   \sum_{j \in J} a_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
   $$

2. **Non-negativity and Integrality**:

   $$
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   $$

#### Parameter Tables

**Shelf Capacities (from capacity.csv, source order):**

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

**Product Values and Weights (from products.csv, source order):**

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

**Complete Mathematical Model:**

$$
\begin{align*}
\max \quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{20} a_j \cdot x_{ij} \leq c_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\end{align*}
$$

Where $c_i$, $v_j$, and $a_j$ are as given in the tables above, and $x_{ij}$ is the number of units of product $j$ to place on shelf $i$.