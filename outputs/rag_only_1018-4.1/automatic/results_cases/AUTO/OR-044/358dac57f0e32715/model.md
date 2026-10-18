Let:
- \( x_{ij} \): integer number of units of product \( j \) to be placed in section \( i \), for all sections \( i \in \{1,2,3,4,5,6,7,8\} \) and products \( j \in \{1,2,3,4,5,6,7,8,9,10\} \).

Parameters (from the data):
- Section capacities (from capacity.csv):

| SectionID | Capacity |
|-----------|----------|
|     1     |   100    |
|     2     |   150    |
|     3     |   120    |
|     4     |   130    |
|     5     |    90    |
|     6     |   110    |
|     7     |   160    |
|     8     |   140    |

- Product values and weights (from products.csv):

| ProductName | Value | Weight |
|-------------|-------|--------|
|      1      |  10   |   2    |
|      2      |  15   |   3    |
|      3      |   8   |   1    |
|      4      |  12   |   2    |
|      5      |  20   |   4    |
|      6      |  25   |   5    |
|      7      |   5   |   1    |
|      8      |  30   |   6    |
|      9      |  18   |   3    |
|     10      |  22   |   4    |

Model:

Variables:
- \( x_{ij} \in \mathbb{Z}_{\geq 0} \) (nonnegative integers), for all \( i = 1,\ldots,8 \), \( j = 1,\ldots,10 \).

Objective:
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} \text{Value}_j \cdot x_{ij}
\]
where \(\text{Value}_j\) is as given above for each product.

Explicitly:
\[
\max \Bigg[
\sum_{i=1}^{8} \Big(
10 x_{i1} + 15 x_{i2} + 8 x_{i3} + 12 x_{i4} + 20 x_{i5} + 25 x_{i6} + 5 x_{i7} + 30 x_{i8} + 18 x_{i9} + 22 x_{i10}
\Big)
\Bigg]
\]

Subject to (for each section \( i \)):
\[
\sum_{j=1}^{10} \text{Weight}_j \cdot x_{ij} \leq \text{Capacity}_i
\]
where \(\text{Weight}_j\) and \(\text{Capacity}_i\) are as given above.

Explicitly, for each section:

Section 1 (\(i=1\), Capacity = 100):
\[
2x_{1,1} + 3x_{1,2} + 1x_{1,3} + 2x_{1,4} + 4x_{1,5} + 5x_{1,6} + 1x_{1,7} + 6x_{1,8} + 3x_{1,9} + 4x_{1,10} \leq 100
\]

Section 2 (\(i=2\), Capacity = 150):
\[
2x_{2,1} + 3x_{2,2} + 1x_{2,3} + 2x_{2,4} + 4x_{2,5} + 5x_{2,6} + 1x_{2,7} + 6x_{2,8} + 3x_{2,9} + 4x_{2,10} \leq 150
\]

Section 3 (\(i=3\), Capacity = 120):
\[
2x_{3,1} + 3x_{3,2} + 1x_{3,3} + 2x_{3,4} + 4x_{3,5} + 5x_{3,6} + 1x_{3,7} + 6x_{3,8} + 3x_{3,9} + 4x_{3,10} \leq 120
\]

Section 4 (\(i=4\), Capacity = 130):
\[
2x_{4,1} + 3x_{4,2} + 1x_{4,3} + 2x_{4,4} + 4x_{4,5} + 5x_{4,6} + 1x_{4,7} + 6x_{4,8} + 3x_{4,9} + 4x_{4,10} \leq 130
\]

Section 5 (\(i=5\), Capacity = 90):
\[
2x_{5,1} + 3x_{5,2} + 1x_{5,3} + 2x_{5,4} + 4x_{5,5} + 5x_{5,6} + 1x_{5,7} + 6x_{5,8} + 3x_{5,9} + 4x_{5,10} \leq 90
\]

Section 6 (\(i=6\), Capacity = 110):
\[
2x_{6,1} + 3x_{6,2} + 1x_{6,3} + 2x_{6,4} + 4x_{6,5} + 5x_{6,6} + 1x_{6,7} + 6x_{6,8} + 3x_{6,9} + 4x_{6,10} \leq 110
\]

Section 7 (\(i=7\), Capacity = 160):
\[
2x_{7,1} + 3x_{7,2} + 1x_{7,3} + 2x_{7,4} + 4x_{7,5} + 5x_{7,6} + 1x_{7,7} + 6x_{7,8} + 3x_{7,9} + 4x_{7,10} \leq 160
\]

Section 8 (\(i=8\), Capacity = 140):
\[
2x_{8,1} + 3x_{8,2} + 1x_{8,3} + 2x_{8,4} + 4x_{8,5} + 5x_{8,6} + 1x_{8,7} + 6x_{8,8} + 3x_{8,9} + 4x_{8,10} \leq 140
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

This model uses all provided data, aligns all indices, and enforces all required constraints and domains as specified.