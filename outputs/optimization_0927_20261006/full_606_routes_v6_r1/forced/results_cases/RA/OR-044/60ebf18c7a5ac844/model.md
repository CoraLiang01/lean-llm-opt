Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and nonnegative.

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

**Decision Variables:**

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all SectionID $i \in \{1,2,3,4,5,6,7,8\}$ and ProductName $j \in \{1,2,3,4,5,6,7,8,9,10\}$

**Objective:**

$$
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ as given above.

**Constraints:**

For each section $i$ (SectionID), the total space used by all products cannot exceed its capacity:

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

where $w_j$ is the Weight (space requirement) of product $j$, and $C_i$ is the Capacity of section $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Explicit Data Used:**

- Section capacities:

  - SectionID 1: 100
  - SectionID 2: 150
  - SectionID 3: 120
  - SectionID 4: 130
  - SectionID 5: 90
  - SectionID 6: 110
  - SectionID 7: 160
  - SectionID 8: 140

- Product values and weights:

  - Product 1: Value 10, Weight 2
  - Product 2: Value 15, Weight 3
  - Product 3: Value 8, Weight 1
  - Product 4: Value 12, Weight 2
  - Product 5: Value 20, Weight 4
  - Product 6: Value 25, Weight 5
  - Product 7: Value 5, Weight 1
  - Product 8: Value 30, Weight 6
  - Product 9: Value 18, Weight 3
  - Product 10: Value 22, Weight 4