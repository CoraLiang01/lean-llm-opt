Let x_{ij} = number of units of product j (ProductName j) to be placed on shelf i (ShelfID i), for i = 1,...,10 and j = 1,...,20. All x_{ij} are nonnegative integers.

Parameters:
- ShelfID and Capacity (from capacity.csv):

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

- ProductName, Value, and Weight (from products.csv):

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

Mathematical Model:

Decision variables:
x_{ij} ∈ {0, 1, 2, ...} for i = 1,...,10 (ShelfID), j = 1,...,20 (ProductName)

Objective:
Maximize total value of products placed on all shelves:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as given above for each ProductName j.

Constraints:
For each shelf i (ShelfID), the total weight of products placed does not exceed its capacity:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i \qquad \forall i = 1,...,10
\]
where Weight_j and Capacity_i are as given above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,...,10;\; j = 1,...,20
\]

Explicitly, for each shelf i (using the data):

For ShelfID 1 (Capacity 500):
\[
10x_{1,1} + 20x_{1,2} + 5x_{1,3} + 15x_{1,4} + 25x_{1,5} + 30x_{1,6} + 12x_{1,7} + 35x_{1,8} + 10x_{1,9} + 20x_{1,10} + 18x_{1,11} + 28x_{1,12} + 8x_{1,13} + 22x_{1,14} + 25x_{1,15} + 40x_{1,16} + 14x_{1,17} + 16x_{1,18} + 50x_{1,19} + 30x_{1,20} \leq 500
\]
Repeat similarly for ShelfID 2 through 10, using their respective capacities.

Objective (fully expanded):
\[
\text{Maximize} \quad
\sum_{i=1}^{10} \Big(
50x_{i,1} + 70x_{i,2} + 30x_{i,3} + 60x_{i,4} + 80x_{i,5} + 90x_{i,6} + 40x_{i,7} + 100x_{i,8} + 55x_{i,9} + 75x_{i,10} + 65x_{i,11} + 95x_{i,12} + 45x_{i,13} + 85x_{i,14} + 70x_{i,15} + 110x_{i,16} + 50x_{i,17} + 60x_{i,18} + 120x_{i,19} + 100x_{i,20}
\Big)
\]

Summary:
- Decision variables: x_{ij} = integer number of units of product j on shelf i, for all i, j.
- Objective: Maximize total value placed.
- Constraints: For each shelf, total weight of products placed ≤ shelf capacity.
- All coefficients and indices are as given in the supplied CSVs.