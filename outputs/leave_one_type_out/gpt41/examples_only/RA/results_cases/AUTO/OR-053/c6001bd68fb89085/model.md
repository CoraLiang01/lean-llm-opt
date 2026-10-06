Let x_{ij} = number of units of product j (ProductName j) to be placed on shelf i (ShelfID i), for i = 1,...,10 and j = 1,...,20. All x_{ij} are nonnegative integers.

Parameters:
- ShelfID and Capacity from capacity.csv:

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

- ProductName, Value, and Weight from products.csv:

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

Decision variables:
x_{ij} ∈ {0, 1, 2, ...} for all ShelfID i ∈ {1,...,10}, ProductName j ∈ {1,...,20}

Objective:
Maximize total value across all shelves:
\[
\text{Maximize} \quad \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as given above for each ProductName j.

Constraints:
For each shelf i ∈ {1,...,10} (using the original order from capacity.csv):

\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where Weight_j is as given above for each ProductName j, and Capacity_i is as given above for each ShelfID i.

Variable domains:
x_{ij} ∈ {0, 1, 2, ...} (integers, nonnegative) for all i, j.

Summary:
- Decision variables: x_{ij} = number of units of product j on shelf i (integers, ≥0)
- Objective: maximize total value placed on all shelves
- Constraints: for each shelf, total weight of products placed does not exceed that shelf’s capacity
- All coefficients and indices are as provided in the CSVs above.