Let x_{ij} = number of units of product j (ProductName j) to be stocked in section i (SectionID i), for i = 1,...,8 and j = 1,...,10. All x_{ij} are nonnegative integers.

Parameters:
- Capacity_i: display space limit of section i (from capacity.csv)
- Value_j: price of product j (from products.csv)
- Weight_j: shelf space required per unit of product j (from products.csv)

Data:
From capacity.csv:
SectionID | Capacity
1         | 100
2         | 150
3         | 120
4         | 130
5         | 90
6         | 110
7         | 160
8         | 140

From products.csv:
ProductName | Value | Weight
1           | 10    | 2
2           | 15    | 3
3           | 8     | 1
4           | 12    | 2
5           | 20    | 4
6           | 25    | 5
7           | 5     | 1
8           | 30    | 6
9           | 18    | 3
10          | 22    | 4

Model:

Decision variables:
x_{ij} ∈ {0, 1, 2, ...} for i = 1,...,8; j = 1,...,10

Objective:
Maximize total revenue:
maximize
 ∑_{i=1}^8 ∑_{j=1}^{10} Value_j * x_{ij}
=
 ∑_{i=1}^8 [10 x_{i1} + 15 x_{i2} + 8 x_{i3} + 12 x_{i4} + 20 x_{i5} + 25 x_{i6} + 5 x_{i7} + 30 x_{i8} + 18 x_{i9} + 22 x_{i10}]

Subject to, for each section i:
 ∑_{j=1}^{10} Weight_j * x_{ij} ≤ Capacity_i

That is, for each section:

Section 1 (Capacity 100):
 2 x_{11} + 3 x_{12} + 1 x_{13} + 2 x_{14} + 4 x_{15} + 5 x_{16} + 1 x_{17} + 6 x_{18} + 3 x_{19} + 4 x_{1,10} ≤ 100

Section 2 (Capacity 150):
 2 x_{21} + 3 x_{22} + 1 x_{23} + 2 x_{24} + 4 x_{25} + 5 x_{26} + 1 x_{27} + 6 x_{28} + 3 x_{29} + 4 x_{2,10} ≤ 150

Section 3 (Capacity 120):
 2 x_{31} + 3 x_{32} + 1 x_{33} + 2 x_{34} + 4 x_{35} + 5 x_{36} + 1 x_{37} + 6 x_{38} + 3 x_{39} + 4 x_{3,10} ≤ 120

Section 4 (Capacity 130):
 2 x_{41} + 3 x_{42} + 1 x_{43} + 2 x_{44} + 4 x_{45} + 5 x_{46} + 1 x_{47} + 6 x_{48} + 3 x_{49} + 4 x_{4,10} ≤ 130

Section 5 (Capacity 90):
 2 x_{51} + 3 x_{52} + 1 x_{53} + 2 x_{54} + 4 x_{55} + 5 x_{56} + 1 x_{57} + 6 x_{58} + 3 x_{59} + 4 x_{5,10} ≤ 90

Section 6 (Capacity 110):
 2 x_{61} + 3 x_{62} + 1 x_{63} + 2 x_{64} + 4 x_{65} + 5 x_{66} + 1 x_{67} + 6 x_{68} + 3 x_{69} + 4 x_{6,10} ≤ 110

Section 7 (Capacity 160):
 2 x_{71} + 3 x_{72} + 1 x_{73} + 2 x_{74} + 4 x_{75} + 5 x_{76} + 1 x_{77} + 6 x_{78} + 3 x_{79} + 4 x_{7,10} ≤ 160

Section 8 (Capacity 140):
 2 x_{81} + 3 x_{82} + 1 x_{83} + 2 x_{84} + 4 x_{85} + 5 x_{86} + 1 x_{87} + 6 x_{88} + 3 x_{89} + 4 x_{8,10} ≤ 140

Variable domains:
x_{ij} ∈ {0, 1, 2, ...} for all i = 1,...,8; j = 1,...,10

This is a complete integer programming formulation using all provided data, maximizing total revenue while respecting each section's display space limit.