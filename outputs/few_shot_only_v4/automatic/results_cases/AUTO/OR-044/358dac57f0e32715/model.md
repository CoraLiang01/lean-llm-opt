Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv) and $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv).

Parameters:
- $v_j$: Value of product $j$ (from Value column in products.csv)
- $w_j$: Weight (space requirement) of product $j$ (from Weight column in products.csv)
- $C_i$: Capacity of section $i$ (from Capacity column in capacity.csv)

Data:

Sections (from capacity.csv):

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

Products (from products.csv):

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

Objective:
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]

Subject to:

Section capacity constraints (for each section $i$):
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \quad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Integrality and non-negativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,3,4,5,6,7,8\},\; j \in \{1,2,3,4,5,6,7,8,9,10\}
\]

Where:
- $v_j$ and $w_j$ are as given in the products.csv table above,
- $C_i$ is as given in the capacity.csv table above.