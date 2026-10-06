Let:
- \( x_{ij} \): Number of units of product \( j \) placed on shelf \( i \), for \( i = 1,\ldots,10 \) (ShelfID from capacity.csv), \( j = 1,\ldots,20 \) (ProductName from products.csv).
- All \( x_{ij} \) are nonnegative integers.

Parameters:
- Shelf capacities (from capacity.csv):

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

- Product values and weights (from products.csv):

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

Model:

Variables:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Objective (maximize total value on all shelves):
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where \(\text{Value}_j\) is as given in products.csv.

Constraints (shelf capacity for each shelf):
For each shelf \( i \) (ShelfID from 1 to 10):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where \(\text{Weight}_j\) is as given in products.csv, and \(\text{Capacity}_i\) is as given in capacity.csv.

Explicitly, for each shelf:

For ShelfID 1:
\[
10x_{1,1} + 20x_{1,2} + 5x_{1,3} + 15x_{1,4} + 25x_{1,5} + 35x_{1,6} + 45x_{1,7} + 55x_{1,8} + 65x_{1,9} + 20x_{1,10} + 18x_{1,11} + 28x_{1,12} + 8x_{1,13} + 28x_{1,14} + 25x_{1,15} + 40x_{1,16} + 55x_{1,17} + 70x_{1,18} + 85x_{1,19} + 100x_{1,20} \leq 750
\]

For ShelfID 2:
\[
10x_{2,1} + 20x_{2,2} + 5x_{2,3} + 15x_{2,4} + 25x_{2,5} + 35x_{2,6} + 45x_{2,7} + 55x_{2,8} + 65x_{2,9} + 20x_{2,10} + 18x_{2,11} + 28x_{2,12} + 8x_{2,13} + 28x_{2,14} + 25x_{2,15} + 40x_{2,16} + 55x_{2,17} + 70x_{2,18} + 85x_{2,19} + 100x_{2,20} \leq 820
\]

... (repeat for ShelfID 3 through 10, using their respective capacities).

Summary:

Maximize
\[
\sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]

Subject to, for each \( i = 1,\ldots,10 \):
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

All coefficients and indices are as given in the CSVs above.