Sets and Indices:
- Let S = {1, 2, ..., 10} be the set of shelves (from capacity.csv, ShelfID).
- Let P = {1, 2, ..., 20} be the set of products (from products.csv, ProductName).
- Let x_{ij} = number of units of product j ∈ P placed on shelf i ∈ S (x_{ij} ∈ {0, 1, 2, ...}).

Parameters:
- Capacity_i: capacity of shelf i (from capacity.csv).
- Value_j: value per unit of product j (from products.csv).
- Weight_j: weight per unit of product j (from products.csv).

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

Mathematical Model:

Variables:
x_{ij} ∈ {0, 1, 2, ...} for all i ∈ S, j ∈ P

Objective:
Maximize total value of products on all shelves:
\[
\text{Maximize} \quad Z = \sum_{i=1}^{10} \sum_{j=1}^{20} \text{Value}_j \cdot x_{ij}
\]
That is,
\[
Z = \sum_{i=1}^{10} \left(
55 x_{i1} + 75 x_{i2} + 65 x_{i3} + 60 x_{i4} + 80 x_{i5} + 90 x_{i6} + 40 x_{i7} + 100 x_{i8} + 55 x_{i9} + 75 x_{i10} + 110 x_{i11} + 50 x_{i12} + 60 x_{i13} + 120 x_{i14} + 70 x_{i15} + 110 x_{i16} + 50 x_{i17} + 60 x_{i18} + 120 x_{i19} + 100 x_{i20}
\right)
\]

Subject to (for each shelf i ∈ S):

Shelf capacity constraints:
\[
\sum_{j=1}^{20} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
That is, for each shelf i:
\[
10 x_{i1} + 20 x_{i2} + 5 x_{i3} + 15 x_{i4} + 25 x_{i5} + 35 x_{i6} + 45 x_{i7} + 55 x_{i8} + 65 x_{i9} + 20 x_{i10} + 18 x_{i11} + 28 x_{i12} + 8 x_{i13} + 28 x_{i14} + 25 x_{i15} + 40 x_{i16} + 55 x_{i17} + 70 x_{i18} + 85 x_{i19} + 100 x_{i20} \leq \text{Capacity}_i
\]
where Capacity_i is as follows:
- Shelf 1: 750
- Shelf 2: 820
- Shelf 3: 570
- Shelf 4: 800
- Shelf 5: 550
- Shelf 6: 900
- Shelf 7: 650
- Shelf 8: 800
- Shelf 9: 850
- Shelf 10: 900

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Summary:
Maximize the total value of products allocated to all shelves, subject to each shelf's capacity, using nonnegative integer variables x_{ij} for the number of units of product j placed on shelf i, with all coefficients and capacities as given above.