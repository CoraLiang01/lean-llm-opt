Let $x_{ij}$ = number of units of product $j$ (item_name $j$) to be placed on shelf $i$ (resource_id $i$), for all $i \in \{1,2,\ldots,10\}$ and $j \in \{1,2,\ldots,20\}$.

Parameters:

- $v_j$ = item_value of product $j$ (from products.csv)
- $w_j$ = resource_requirement of product $j$ (from products.csv)
- $C_i$ = resource_capacity of shelf $i$ (from capacity.csv)

Data:

capacity.csv

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

products.csv

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
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

Subject to, for each shelf $i$ (resource_id):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\ j \in \{1,2,\ldots,20\}
\]

Where:

- $v_j$ and $w_j$ are as given in products.csv for item_name $j$.
- $C_i$ is as given in capacity.csv for resource_id $i$.
- All indices and coefficients are as shown in the tables above, preserving original file and row order.