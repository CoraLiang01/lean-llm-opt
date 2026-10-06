Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $\geq 0$.

Indices:
- $i$ indexes SectionID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes ProductName from products.csv: $j \in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters:
- $c_i$ = Capacity of section $i$ (from capacity.csv)
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (shelf space requirement) of product $j$ (from products.csv)

Data:

Section capacities ($c_i$):
- $c_1 = 100$
- $c_2 = 150$
- $c_3 = 120$
- $c_4 = 130$
- $c_5 = 90$
- $c_6 = 110$
- $c_7 = 160$
- $c_8 = 140$

Product values ($v_j$) and weights ($w_j$):

| ProductName ($j$) | $v_j$ | $w_j$ |
|-------------------|-------|-------|
| 1                 | 10    | 2     |
| 2                 | 15    | 3     |
| 3                 | 8     | 1     |
| 4                 | 12    | 2     |
| 5                 | 20    | 4     |
| 6                 | 25    | 5     |
| 7                 | 5     | 1     |
| 8                 | 30    | 6     |
| 9                 | 18    | 3     |
| 10                | 22    | 4     |

Model:

Objective:
$$
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j x_{ij}
$$

Subject to, for each section $i$:
$$
\sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,8\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

Where:
- $v_j$ and $w_j$ are as given above for each product $j$.
- $c_i$ is as given above for each section $i$.