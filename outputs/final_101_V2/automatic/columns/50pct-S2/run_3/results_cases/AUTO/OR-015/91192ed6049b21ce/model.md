Let $x_{ij}$ = number of units of product $j$ to be placed on shelf $i$, for all shelves $i$ and products $j$.

Indices:
- $i$ indexes shelves, with resource_id from 1 to 10.
- $j$ indexes products, with item_name from 1 to 20.

Parameters:
- $v_j$ = item_value of product $j$
- $w_j$ = resource_requirement (weight) of product $j$
- $C_i$ = resource_capacity of shelf $i$

Data:

From capacity.csv:

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

From products.csv:

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

Mathematical Model:

Objective:
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$

Subject to, for each shelf $i$ (resource_id):

Capacity constraints:
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \quad \forall i = 1,\ldots,10
$$

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
$$

Where:
- $v_j$ and $w_j$ are as given above for each item_name $j$.
- $C_i$ is as given above for each resource_id $i$.

All variables and parameters use the exact identifiers and values as retrieved.