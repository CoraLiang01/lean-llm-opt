Let $x_{ij}$ be the number of units of product $j$ (with item_name $j$) to be placed on shelf $i$ (with resource_id $i$). All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Shelves (resource_id): 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
- Shelf capacities (resource_capacity):

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

- Products (item_name): 1, 2, ..., 20
- Product values (item_value) and weights (resource_requirement):

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

**Mathematical Model:**

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
$$

**Objective:**
$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
$$
where $v_j$ is the item_value of product $j$.

**Constraints:**

For each shelf $i$ (resource_id $i$):
$$
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,10\}
$$
where $w_j$ is the resource_requirement of product $j$, and $c_i$ is the resource_capacity of shelf $i$.

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**All data used:**

capacity.csv

| previous_period_capacity | resource_id | resource_capacity |
|-------------------------|-------------|------------------|
| 564                     | 1           | 500              |
| 679                     | 2           | 700              |
| 481                     | 3           | 600              |
| 684                     | 4           | 800              |
| 623                     | 5           | 550              |
| 1019                    | 6           | 900              |
| 671                     | 7           | 650              |
| 761                     | 8           | 750              |
| 951                     | 9           | 820              |
| 522                     | 10          | 570              |

products.csv

| previous_period_stock_status | item_name | item_value | resource_requirement | previous_period_unit_value |
|-----------------------------|-----------|------------|---------------------|---------------------------|
| Stockout                    | 1         | 50         | 10                  | 48                        |
| Stockout                    | 2         | 70         | 20                  | 56                        |
| Stockout                    | 3         | 30         | 5                   | 31                        |
| Stockout                    | 4         | 60         | 15                  | 53                        |
| Overstock                   | 5         | 80         | 25                  | 96                        |
| Overstock                   | 6         | 90         | 30                  | 91                        |
| Stockout                    | 7         | 40         | 12                  | 45                        |
| Stockout                    | 8         | 100        | 35                  | 119                       |
| Stockout                    | 9         | 55         | 10                  | 62                        |
| Balanced                    | 10        | 75         | 20                  | 60                        |
| Stockout                    | 11        | 65         | 18                  | 67                        |
| Balanced                    | 12        | 95         | 28                  | 84                        |
| Balanced                    | 13        | 45         | 8                   | 38                        |
| Balanced                    | 14        | 85         | 22                  | 68                        |
| Balanced                    | 15        | 70         | 25                  | 74                        |
| Balanced                    | 16        | 110        | 40                  | 119                       |
| Balanced                    | 17        | 50         | 14                  | 42                        |
| Overstock                   | 18        | 60         | 16                  | 48                        |
| Overstock                   | 19        | 120        | 50                  | 117                       |
| Overstock                   | 20        | 100        | 30                  | 93                        |