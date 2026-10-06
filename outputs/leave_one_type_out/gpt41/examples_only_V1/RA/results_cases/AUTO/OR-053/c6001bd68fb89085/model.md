Let x_{ij} = number of units of product j (ProductName j from products.csv) to be placed on shelf i (ShelfID i from capacity.csv), for i = 1,...,10 and j = 1,...,20.

Decision variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ {1,...,10}, j ∈ {1,...,20}

Objective:
Maximize total value of products placed on all shelves:
maximize
∑_{i=1}^{10} ∑_{j=1}^{20} Value_j * x_{ij}

where Value_j is as follows (from products.csv):

ProductName | Value
--- | ---
1 | 50
2 | 70
3 | 30
4 | 60
5 | 80
6 | 90
7 | 40
8 | 100
9 | 55
10 | 75
11 | 65
12 | 95
13 | 45
14 | 85
15 | 70
16 | 110
17 | 50
18 | 60
19 | 120
20 | 100

Subject to:

For each shelf i ∈ {1,...,10}, the total weight of products on shelf i does not exceed its capacity:

∑_{j=1}^{20} Weight_j * x_{ij} ≤ Capacity_i  for all i = 1,...,10

where Weight_j and Capacity_i are as follows:

ProductName | Weight
--- | ---
1 | 10
2 | 20
3 | 5
4 | 15
5 | 25
6 | 30
7 | 12
8 | 35
9 | 10
10 | 20
11 | 18
12 | 28
13 | 8
14 | 22
15 | 25
16 | 40
17 | 14
18 | 16
19 | 50
20 | 30

ShelfID | Capacity
--- | ---
1 | 500
2 | 700
3 | 600
4 | 800
5 | 550
6 | 900
7 | 650
8 | 750
9 | 820
10 | 570

Variable domains:
x_{ij} ∈ {0, 1, 2, ...} (integers, nonnegative) for all i, j

Summary:
maximize ∑_{i=1}^{10} ∑_{j=1}^{20} Value_j * x_{ij}
subject to ∑_{j=1}^{20} Weight_j * x_{ij} ≤ Capacity_i for all i = 1,...,10
      x_{ij} ∈ {0, 1, 2, ...} for all i, j

All coefficients and identifiers are as given in the CSVs above.