Let:
- ShelfID i ∈ {1,2,3,4,5,6,7,8,9,10} (from capacity.csv)
- Product j ∈ {1,2,...,20} (from products.csv)
- x_{ij} = number of units of product j placed on shelf i (integer, x_{ij} ≥ 0)

Parameters:
- Capacity_i: capacity of shelf i (from capacity.csv)
- Value_j: value of product j (from products.csv)
- Weight_j: weight of product j (from products.csv)

Data:

capacity.csv

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

products.csv

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

Variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ {1,...,10}, j ∈ {1,...,20}

Objective:
Maximize total value across all shelves:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
where Value_j is as given above for each product j.

Constraints:
For each shelf i ∈ {1,...,10}:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where Weight_j and Capacity_i are as given above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,...,10\},\ j \in \{1,...,20\}
\]

All coefficients and identifiers are as provided in the CSVs. No additional constraints or data are assumed.