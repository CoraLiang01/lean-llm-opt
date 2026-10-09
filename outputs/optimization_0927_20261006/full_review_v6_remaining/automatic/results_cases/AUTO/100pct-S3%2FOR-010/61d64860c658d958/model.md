Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $\geq 0$.

**Indices:**
- $i \in \{\text{SectionID }1,2,3,4,5,6,7,8\}$
- $j \in \{\text{ProductName }1,2,3,4,5,6,7,8,9,10\}$

**Parameters:**
- $v_j$: Value of product $j$ (from "Value" column)
- $w_j$: Space requirement of product $j$ (from "Weight" column)
- $C_i$: Capacity of section $i$ (from "Capacity" column)

**Data (in source order):**

Sections (from capacity.csv):

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

Products (from products.csv):

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

---

**Mathematical Model:**

Maximize total revenue:
$$
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \cdot x_{ij}
$$

Subject to, for each section $i$:
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

Integrality and nonnegativity:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Where:**

- $v_j$ and $w_j$ are as given above for each product $j$.
- $C_i$ is as given above for each section $i$.

**All data and indices are preserved in source order as required.**