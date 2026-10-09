Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i$ indexes sections, with SectionID from the capacity.csv file.
- $j$ indexes products, with ProductName from the products.csv file.

**Parameters:**
- $c_i$: Capacity of section $i$ (from "Capacity" in capacity.csv).
- $v_j$: Value (price) of product $j$ (from "Value" in products.csv).
- $w_j$: Space requirement (shelf space) of product $j$ (from "Weight" in products.csv).

**Data:**

Sections (from capacity.csv, in file order):

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

Products (from products.csv, in file order):

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

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j\, x_{ij}
$$

**Subject to:**

For each section $i$ (SectionID as below):

$$
\sum_{j=1}^{10} w_j\, x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

That is, explicitly:

- For SectionID 1: $\sum_{j=1}^{10} w_j\, x_{1j} \leq 100$
- For SectionID 2: $\sum_{j=1}^{10} w_j\, x_{2j} \leq 150$
- For SectionID 3: $\sum_{j=1}^{10} w_j\, x_{3j} \leq 120$
- For SectionID 4: $\sum_{j=1}^{10} w_j\, x_{4j} \leq 130$
- For SectionID 5: $\sum_{j=1}^{10} w_j\, x_{5j} \leq 90$
- For SectionID 6: $\sum_{j=1}^{10} w_j\, x_{6j} \leq 110$
- For SectionID 7: $\sum_{j=1}^{10} w_j\, x_{7j} \leq 160$
- For SectionID 8: $\sum_{j=1}^{10} w_j\, x_{8j} \leq 140$

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Where:**

- $v_j$ and $w_j$ for each product $j$ are:

| $j$ | ProductName | $v_j$ | $w_j$ |
|-----|-------------|-------|-------|
| 1   | 1           | 10    | 2     |
| 2   | 2           | 15    | 3     |
| 3   | 3           | 8     | 1     |
| 4   | 4           | 12    | 2     |
| 5   | 5           | 20    | 4     |
| 6   | 6           | 25    | 5     |
| 7   | 7           | 5     | 1     |
| 8   | 8           | 30    | 6     |
| 9   | 9           | 18    | 3     |
| 10  | 10          | 22    | 4     |

- $c_i$ for each section $i$ is as above.

---

**Summary:**  
Maximize total revenue from all stocked products across all sections, subject to each section's display space limit. All product allocations are nonnegative integers.