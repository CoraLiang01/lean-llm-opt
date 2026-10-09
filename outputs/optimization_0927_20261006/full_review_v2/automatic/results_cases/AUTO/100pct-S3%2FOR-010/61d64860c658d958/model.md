Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and nonnegative.

**Indices:**
- $i$ indexes sections, with SectionID from 1 to 8 (in source order).
- $j$ indexes products, with ProductName from 1 to 10 (in source order).

**Parameters:**
- $p_j$: Value of product $j$ (from "Value" in products.csv)
- $w_j$: Space requirement of product $j$ (from "Weight" in products.csv)
- $C_i$: Capacity of section $i$ (from "Capacity" in capacity.csv)

**Data (source order):**

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

### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**
$$
\max \sum_{i=1}^8 \sum_{j=1}^{10} p_j\, x_{ij}
$$
where $p_j$ is the Value of product $j$.

**Constraints:**

For each section $i$ (SectionID as below):

$$
\sum_{j=1}^{10} w_j\, x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

where $w_j$ is the Weight (space requirement) of product $j$, and $C_i$ is the Capacity of section $i$.

**Explicitly, for each section:**

- Section 1: $\sum_{j=1}^{10} w_j\, x_{1j} \leq 100$
- Section 2: $\sum_{j=1}^{10} w_j\, x_{2j} \leq 150$
- Section 3: $\sum_{j=1}^{10} w_j\, x_{3j} \leq 120$
- Section 4: $\sum_{j=1}^{10} w_j\, x_{4j} \leq 130$
- Section 5: $\sum_{j=1}^{10} w_j\, x_{5j} \leq 90$
- Section 6: $\sum_{j=1}^{10} w_j\, x_{6j} \leq 110$
- Section 7: $\sum_{j=1}^{10} w_j\, x_{7j} \leq 160$
- Section 8: $\sum_{j=1}^{10} w_j\, x_{8j} \leq 140$

**Variable domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**Summary of Parameters (source order):**

- Sections: SectionID = 1, 2, 3, 4, 5, 6, 7, 8; Capacities = 100, 150, 120, 130, 90, 110, 160, 140
- Products: ProductName = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10; Values = 10, 15, 8, 12, 20, 25, 5, 30, 18, 22; Weights = 2, 3, 1, 2, 4, 5, 1, 6, 3, 4

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^8 \sum_{j=1}^{10} p_j\, x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{10} w_j\, x_{ij} \leq C_i \qquad \forall i = 1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,8,\ j = 1,\ldots,10
\end{align*}
$$

with all parameters as listed above, in the original source order.