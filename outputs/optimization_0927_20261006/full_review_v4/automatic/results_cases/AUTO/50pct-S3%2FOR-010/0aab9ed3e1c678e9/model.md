Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv)
- $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv)

**Parameters:**
- $v_j$: Value of product $j$ (from "Value" in products.csv)
- $w_j$: Weight (space requirement) of product $j$ (from "Weight" in products.csv)
- $C_i$: Capacity of section $i$ (from "Capacity" in capacity.csv)

**Data:**

From capacity.csv (in source order):

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

From products.csv (in source order):

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
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

**Subject to:**

For each section $i$ (SectionID as below):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

That is, explicitly:

- $\sum_{j=1}^{10} w_j \cdot x_{1j} \leq 100$
- $\sum_{j=1}^{10} w_j \cdot x_{2j} \leq 150$
- $\sum_{j=1}^{10} w_j \cdot x_{3j} \leq 120$
- $\sum_{j=1}^{10} w_j \cdot x_{4j} \leq 130$
- $\sum_{j=1}^{10} w_j \cdot x_{5j} \leq 90$
- $\sum_{j=1}^{10} w_j \cdot x_{6j} \leq 110$
- $\sum_{j=1}^{10} w_j \cdot x_{7j} \leq 160$
- $\sum_{j=1}^{10} w_j \cdot x_{8j} \leq 140$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Parameter values:**

- $v_1 = 10$, $w_1 = 2$
- $v_2 = 15$, $w_2 = 3$
- $v_3 = 8$, $w_3 = 1$
- $v_4 = 12$, $w_4 = 2$
- $v_5 = 20$, $w_5 = 4$
- $v_6 = 25$, $w_6 = 5$
- $v_7 = 5$, $w_7 = 1$
- $v_8 = 30$, $w_8 = 6$
- $v_9 = 18$, $w_9 = 3$
- $v_{10} = 22$, $w_{10} = 4$

- $C_1 = 100$
- $C_2 = 150$
- $C_3 = 120$
- $C_4 = 130$
- $C_5 = 90$
- $C_6 = 110$
- $C_7 = 160$
- $C_8 = 140$

---

**Summary:**  
Maximize total revenue from product allocations, subject to section capacity constraints, with integer nonnegative decision variables for each product-section pair. All coefficients and identifiers are as retrieved and in original order.