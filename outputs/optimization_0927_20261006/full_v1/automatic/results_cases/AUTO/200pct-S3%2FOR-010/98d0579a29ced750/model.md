Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $\geq 0$.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv)
- $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of section $i$ (from "Capacity" in capacity.csv)
- $p_j$ = Value of product $j$ (from "Value" in products.csv)
- $w_j$ = Weight (shelf space requirement) of product $j$ (from "Weight" in products.csv)

**Data:**

Sections (from capacity.csv, in source order):

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

Products (from products.csv, in source order):

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

### Mathematical Model

**Objective:**

$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} p_j \cdot x_{ij}
$$

**Subject to:**

For each section $i$ (SectionID as below):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

That is, explicitly:

- $\sum_{j=1}^{10} w_j x_{1j} \leq 100$
- $\sum_{j=1}^{10} w_j x_{2j} \leq 150$
- $\sum_{j=1}^{10} w_j x_{3j} \leq 120$
- $\sum_{j=1}^{10} w_j x_{4j} \leq 130$
- $\sum_{j=1}^{10} w_j x_{5j} \leq 90$
- $\sum_{j=1}^{10} w_j x_{6j} \leq 110$
- $\sum_{j=1}^{10} w_j x_{7j} \leq 160$
- $\sum_{j=1}^{10} w_j x_{8j} \leq 140$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Where:**

- $p_j$ and $w_j$ are as given in the table above, matched by ProductName.
- $c_i$ is as given in the table above, matched by SectionID.

---

**All data and identifiers are preserved in source order as required.**