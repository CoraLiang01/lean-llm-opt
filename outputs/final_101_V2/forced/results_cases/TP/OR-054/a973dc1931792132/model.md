##### Decision Variables

$x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$, for each shelf $i \in S$ and product $j \in P$.

##### Parameters

- $S = \{1,2,3,4,5,6,7,8,9,10\}$ (Shelf IDs)
- $P = \{1,2,\ldots,20\}$ (Product Names)
- Shelf capacities $C_i$:

| ShelfID | Capacity |
|---------|----------|
| 1       | 750      |
| 2       | 820      |
| 3       | 570      |
| 4       | 800      |
| 5       | 550      |
| 6       | 900      |
| 7       | 650      |
| 8       | 800      |
| 9       | 850      |
| 10      | 900      |

- Product values $v_j$ and weights $w_j$:

| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 55    | 10     |
| 2           | 75    | 20     |
| 3           | 65    | 5      |
| 4           | 60    | 15     |
| 5           | 80    | 25     |
| 6           | 90    | 35     |
| 7           | 40    | 45     |
| 8           | 100   | 55     |
| 9           | 55    | 65     |
| 10          | 75    | 20     |
| 11          | 110   | 18     |
| 12          | 50    | 28     |
| 13          | 60    | 8      |
| 14          | 120   | 28     |
| 15          | 70    | 25     |
| 16          | 110   | 40     |
| 17          | 50    | 55     |
| 18          | 60    | 70     |
| 19          | 120   | 85     |
| 20          | 100   | 100    |

##### Objective Function

$$
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
$$

##### Constraints

For each shelf $i=1,\ldots,10$:
$$
\sum_{j=1}^{20} w_j x_{ij} \leq C_i
$$

For all $i=1,\ldots,10$, $j=1,\ldots,20$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $C_i$, $v_j$, and $w_j$ are as listed above.