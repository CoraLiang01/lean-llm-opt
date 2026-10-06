Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i$ indexes shelves (resource_id from capacity.csv) and $j$ indexes products (item_name from products.csv). All $x_{ij}$ are nonnegative integers.

Parameters:

- $c_i$: capacity of shelf $i$ (resource_capacity from capacity.csv)
- $v_j$: value of product $j$ (item_value from products.csv)
- $w_j$: weight of product $j$ (resource_requirement from products.csv)

Data:

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

Mathematical Model:

Objective:
$$
\max \sum_{i \in \{1,\ldots,10\}} \sum_{j \in \{1,\ldots,20\}} v_j \cdot x_{ij}
$$

Subject to, for each shelf $i$ (resource_id):

$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

Where:

- $c_i$ is resource_capacity for shelf $i$ (from capacity.csv)
- $v_j$ is item_value for product $j$ (from products.csv)
- $w_j$ is resource_requirement for product $j$ (from products.csv)
- $x_{ij}$ is the number of units of product $j$ placed on shelf $i$ (integer, $\geq 0$)