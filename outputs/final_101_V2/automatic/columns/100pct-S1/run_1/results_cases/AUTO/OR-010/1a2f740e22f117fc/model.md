Let $x_{ij}$ = number of units of product $j$ to be placed in section $i$.

Indices:
- $i$ indexes SectionID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes ProductName from products.csv: $j \in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters:
- $c_i$ = Capacity of section $i$ (from capacity.csv)
- $p_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (shelf space requirement) of product $j$ (from products.csv)

Data:

Section Capacities (from capacity.csv):

| SectionID | Capacity |
|-----------|----------|
| 1         | 100      |
| 2         | 150      |
| 3         | 120      |
| 4         | 130      |
| 5         | 90       |
| 6         | 110      |
| 7         | 160      |
| 8         | 140      |

Product Data (from products.csv):

| ProductName | Value | Weight |
|-------------|-------|--------|
| 1           | 10    | 2      |
| 2           | 15    | 3      |
| 3           | 8     | 1      |
| 4           | 12    | 2      |
| 5           | 20    | 4      |
| 6           | 25    | 5      |
| 7           | 5     | 1      |
| 8           | 30    | 6      |
| 9           | 18    | 3      |
| 10          | 22    | 4      |

Mathematical Model:

Objective:
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} p_j \cdot x_{ij}
$$

Subject to (for each section $i$):

Capacity constraints:
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,3,4,5,6,7,8\},\; j \in \{1,2,3,4,5,6,7,8,9,10\}
$$

Where:
- $p_j$ and $w_j$ are as given in the table above for each product $j$.
- $c_i$ is as given in the table above for each section $i$.