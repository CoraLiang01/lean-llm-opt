Let:
- S = {1,2,3,4,5,6,7,8,9,10} (ShelfID from capacity.csv)
- P = {1,2,...,20} (ProductName from products.csv)
- x_{ij} = number of units of product j placed on shelf i, for i in S, j in P (x_{ij} ∈ {0,1,2,...})

Parameters:
- Capacity_i: capacity of shelf i (from capacity.csv)
- Value_j: value per unit of product j (from products.csv)
- Weight_j: weight per unit of product j (from products.csv)

Data:
From capacity.csv:
ShelfID | Capacity
------- | -------
1 | 750
2 | 820
3 | 570
4 | 800
5 | 550
6 | 900
7 | 650
8 | 800
9 | 850
10 | 900

From products.csv:
ProductName | Value | Weight
----------- | ----- | ------
1 | 55 | 10
2 | 75 | 20
3 | 65 | 5
4 | 60 | 15
5 | 80 | 25
6 | 90 | 35
7 | 40 | 45
8 | 100 | 55
9 | 55 | 65
10 | 75 | 20
11 | 110 | 18
12 | 50 | 28
13 | 60 | 8
14 | 120 | 28
15 | 70 | 25
16 | 110 | 40
17 | 50 | 55
18 | 60 | 70
19 | 120 | 85
20 | 100 | 100

Model:

Variables:
x_{ij} ∈ {0, 1, 2, ...} for i ∈ {1,...,10}, j ∈ {1,...,20}

Objective:
Maximize ∑_{i=1}^{10} ∑_{j=1}^{20} Value_j * x_{ij}
  =  ∑_{i=1}^{10} [55 x_{i1} + 75 x_{i2} + 65 x_{i3} + 60 x_{i4} + 80 x_{i5} + 90 x_{i6} + 40 x_{i7} + 100 x_{i8} + 55 x_{i9} + 75 x_{i10} + 110 x_{i11} + 50 x_{i12} + 60 x_{i13} + 120 x_{i14} + 70 x_{i15} + 110 x_{i16} + 50 x_{i17} + 60 x_{i18} + 120 x_{i19} + 100 x_{i20}]

Subject to (for each shelf i):

  ∑_{j=1}^{20} Weight_j * x_{ij} ≤ Capacity_i

That is, for each shelf:

Shelf 1: 10 x_{1,1} + 20 x_{1,2} + 5 x_{1,3} + 15 x_{1,4} + 25 x_{1,5} + 35 x_{1,6} + 45 x_{1,7} + 55 x_{1,8} + 65 x_{1,9} + 20 x_{1,10} + 18 x_{1,11} + 28 x_{1,12} + 8 x_{1,13} + 28 x_{1,14} + 25 x_{1,15} + 40 x_{1,16} + 55 x_{1,17} + 70 x_{1,18} + 85 x_{1,19} + 100 x_{1,20} ≤ 750

Shelf 2: 10 x_{2,1} + 20 x_{2,2} + ... + 100 x_{2,20} ≤ 820

Shelf 3: 10 x_{3,1} + 20 x_{3,2} + ... + 100 x_{3,20} ≤ 570

Shelf 4: 10 x_{4,1} + 20 x_{4,2} + ... + 100 x_{4,20} ≤ 800

Shelf 5: 10 x_{5,1} + 20 x_{5,2} + ... + 100 x_{5,20} ≤ 550

Shelf 6: 10 x_{6,1} + 20 x_{6,2} + ... + 100 x_{6,20} ≤ 900

Shelf 7: 10 x_{7,1} + 20 x_{7,2} + ... + 100 x_{7,20} ≤ 650

Shelf 8: 10 x_{8,1} + 20 x_{8,2} + ... + 100 x_{8,20} ≤ 800

Shelf 9: 10 x_{9,1} + 20 x_{9,2} + ... + 100 x_{9,20} ≤ 850

Shelf 10: 10 x_{10,1} + 20 x_{10,2} + ... + 100 x_{10,20} ≤ 900

And for all i, j: x_{ij} ∈ {0, 1, 2, ...}

This is a complete integer programming formulation for the BigMart shelf allocation problem using the provided data.