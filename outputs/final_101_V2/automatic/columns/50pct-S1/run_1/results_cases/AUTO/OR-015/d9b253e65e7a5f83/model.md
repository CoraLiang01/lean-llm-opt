Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

Indices:
- $i$: shelf (resource_id from capacity.csv)
- $j$: product (item_name from products.csv)

Parameters:
- $v_j$: item_value of product $j$ (from products.csv)
- $w_j$: resource_requirement of product $j$ (from products.csv)
- $C_i$: resource_capacity of shelf $i$ (from capacity.csv)

Data:

capacity.csv

| archive_revision_number | resource_id | resource_capacity |
|------------------------|-------------|------------------|
| 5                      | 1           | 500              |
| 8                      | 2           | 700              |
| 1                      | 3           | 600              |
| 2                      | 4           | 800              |
| 2                      | 5           | 550              |
| 7                      | 6           | 900              |
| 4                      | 7           | 650              |
| 1                      | 8           | 750              |
| 2                      | 9           | 820              |
| 2                      | 10          | 570              |

products.csv

| record_keeper_group | item_name | item_value | resource_requirement | archive_revision_number |
|---------------------|-----------|------------|----------------------|------------------------|
| Team B              | 1         | 50         | 10                   | 4                      |
| Team A              | 2         | 70         | 20                   | 3                      |
| Team C              | 3         | 30         | 5                    | 8                      |
| Team B              | 4         | 60         | 15                   | 5                      |
| Team B              | 5         | 80         | 25                   | 2                      |
| Team C              | 6         | 90         | 30                   | 1                      |
| Team B              | 7         | 40         | 12                   | 5                      |
| Team B              | 8         | 100        | 35                   | 8                      |
| Team C              | 9         | 55         | 10                   | 2                      |
| Team C              | 10        | 75         | 20                   | 6                      |
| Team C              | 11        | 65         | 18                   | 8                      |
| Team B              | 12        | 95         | 28                   | 3                      |
| Team A              | 13        | 45         | 8                    | 8                      |
| Team A              | 14        | 85         | 22                   | 4                      |
| Team B              | 15        | 70         | 25                   | 3                      |
| Team A              | 16        | 110        | 40                   | 3                      |
| Team A              | 17        | 50         | 14                   | 2                      |
| Team B              | 18        | 60         | 16                   | 5                      |
| Team C              | 19        | 120        | 50                   | 8                      |
| Team B              | 20        | 100        | 30                   | 8                      |

Mathematical Model:

Objective:
\[
\max \sum_{i \in \{1,2,\ldots,10\}} \sum_{j \in \{1,2,\ldots,20\}} v_j \cdot x_{ij}
\]

Subject to, for each shelf $i$ (resource_id):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\ j \in \{1,2,\ldots,20\}
\]

Where:
- $v_j$ and $w_j$ are taken from the item_value and resource_requirement columns for each item_name $j$ in products.csv.
- $C_i$ is the resource_capacity for each resource_id $i$ in capacity.csv.

All indices, coefficients, and constraints use the exact identifiers and values as retrieved.