Let $x_{ij}$ be the number of units of product $j$ to be placed on shelf $i$, where $i \in \{1,2,3,4,5,6,7,8,9,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

Parameters:

- $v_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight of product $j$ (from products.csv)
- $C_i$: Capacity of shelf $i$ (from capacity.csv)

Data:

From capacity.csv (in source order):

| ShelfID | Capacity |
|---------|----------|
| 1       | 500      |
| 2       | 700      |
| 3       | 600      |
| 4       | 800      |
| 5       | 550      |
| 6       | 900      |
| 7       | 650      |
| 8       | 750      |
| 9       | 820      |
| 10      | 570      |

From products.csv (in source order):

| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 50    | 10     |
| 2           | 70    | 20     |
| 3           | 30    | 5      |
| 4           | 60    | 15     |
| 5           | 80    | 25     |
| 6           | 90    | 30     |
| 7           | 40    | 12     |
| 8           | 100   | 35     |
| 9           | 55    | 10     |
| 10          | 75    | 20     |
| 11          | 65    | 18     |
| 12          | 95    | 28     |
| 13          | 45    | 8      |
| 14          | 85    | 22     |
| 15          | 70    | 25     |
| 16          | 110   | 40     |
| 17          | 50    | 14     |
| 18          | 60    | 16     |
| 19          | 120   | 50     |
| 20          | 100   | 30     |

Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]

Subject to, for each shelf $i$:

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Where:

- $v_j$ and $w_j$ are as given in the products.csv table above, matched by ProductName.
- $C_i$ is as given in the capacity.csv table above, matched by ShelfID.

All data and indices are preserved in original file and row order.