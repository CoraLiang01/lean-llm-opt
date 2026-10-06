Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

**Parameters:**

- Sections (from capacity.csv):

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

- Products (from products.csv):

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

**Mathematical Model:**

Objective:
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

Subject to, for each section $i$:
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,8
$$
where $w_j$ is the Weight (space requirement) of product $j$, and $C_i$ is the Capacity of section $i$.

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,8;\ j = 1,\ldots,10
$$

**Where:**

- $i$ indexes SectionID: 1, 2, ..., 8
- $j$ indexes ProductName: 1, 2, ..., 10
- $v_j$ and $w_j$ are as given in the table above
- $C_i$ is the Capacity for section $i$ as given above

**All coefficients and identifiers are as retrieved.**