Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv)
- $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv)

**Parameters:**
- $p_j$: Value of product $j$ (from "Value" in products.csv)
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
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} p_j \cdot x_{ij}
\]
where $p_j$ is as above.

**Subject to:**

For each section $i$:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]
where $w_j$ and $C_i$ are as above.

**Variable domains:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

---

**Explicit Data Table for Parameters:**

- $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$
- $(p_j, w_j)$ for $j=1$ to $10$:

| $j$ | $p_j$ | $w_j$ |
|-----|-------|-------|
| 1   | 10    | 2     |
| 2   | 15    | 3     |
| 3   | 8     | 1     |
| 4   | 12    | 2     |
| 5   | 20    | 4     |
| 6   | 25    | 5     |
| 7   | 5     | 1     |
| 8   | 30    | 6     |
| 9   | 18    | 3     |
| 10  | 22    | 4     |

---

**Summary:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{8} \sum_{j=1}^{10} p_j x_{ij} \\
\text{s.t. } & \sum_{j=1}^{10} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,8;\ j=1,\ldots,10
\end{align*}
\]

with all coefficients and identifiers as above.