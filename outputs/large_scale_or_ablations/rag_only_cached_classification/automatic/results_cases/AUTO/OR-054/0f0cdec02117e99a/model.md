Let:
- \( S \) be the set of shelves, indexed by \( i \) (from ShelfID in capacity.csv: \( i = 1,2,\ldots,10 \))
- \( P \) be the set of products, indexed by \( j \) (from ProductName in products.csv: \( j = 1,2,\ldots,20 \))
- \( x_{ij} \) = number of units of product \( j \) placed on shelf \( i \) (decision variable, integer, \( x_{ij} \geq 0 \))

Parameters:
- \( C_i \) = capacity of shelf \( i \) (from capacity.csv)
- \( v_j \) = value per unit of product \( j \) (from products.csv)
- \( w_j \) = weight per unit of product \( j \) (from products.csv)

Data:

capacity.csv

| ShelfID (\( i \)) | Capacity (\( C_i \)) |
|-------------------|----------------------|
| 1                 | 750                  |
| 2                 | 820                  |
| 3                 | 570                  |
| 4                 | 800                  |
| 5                 | 550                  |
| 6                 | 900                  |
| 7                 | 650                  |
| 8                 | 800                  |
| 9                 | 850                  |
| 10                | 900                  |

products.csv

| ProductName (\( j \)) | Value (\( v_j \)) | Weight (\( w_j \)) |
|-----------------------|-------------------|--------------------|
| 1                     | 55                | 10                 |
| 2                     | 75                | 20                 |
| 3                     | 65                | 5                  |
| 4                     | 60                | 15                 |
| 5                     | 80                | 25                 |
| 6                     | 90                | 35                 |
| 7                     | 40                | 45                 |
| 8                     | 100               | 55                 |
| 9                     | 55                | 65                 |
| 10                    | 75                | 20                 |
| 11                    | 110               | 18                 |
| 12                    | 50                | 28                 |
| 13                    | 60                | 8                  |
| 14                    | 120               | 28                 |
| 15                    | 70                | 25                 |
| 16                    | 110               | 40                 |
| 17                    | 50                | 55                 |
| 18                    | 60                | 70                 |
| 19                    | 120               | 85                 |
| 20                    | 100               | 100                |

Mathematical Optimization Model:

Variables:
- \( x_{ij} \in \mathbb{Z}_{\geq 0} \) for all \( i \in \{1,\ldots,10\} \), \( j \in \{1,\ldots,20\} \)

Objective:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]
where \( v_j \) is as above.

Subject to (for each shelf \( i \)):
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,10\}
\]
where \( w_j \) and \( C_i \) are as above.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

All coefficients and indices are explicitly from the provided data. No additional constraints or data are assumed.