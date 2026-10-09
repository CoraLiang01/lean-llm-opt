Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

**Indices:**
- $i$ indexes sections, with SectionID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes products, with ProductName from products.csv: $j \in \{1,2,3,4,5,6,7,8,9,10\}$

**Parameters:**
- $c_i$ = Capacity of section $i$ (from capacity.csv)
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (shelf space requirement) of product $j$ (from products.csv)

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
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each section $i$ (SectionID as below):

\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

That is, explicitly:

- For SectionID 1: $\sum_{j=1}^{10} w_j x_{1j} \leq 100$
- For SectionID 2: $\sum_{j=1}^{10} w_j x_{2j} \leq 150$
- For SectionID 3: $\sum_{j=1}^{10} w_j x_{3j} \leq 120$
- For SectionID 4: $\sum_{j=1}^{10} w_j x_{4j} \leq 130$
- For SectionID 5: $\sum_{j=1}^{10} w_j x_{5j} \leq 90$
- For SectionID 6: $\sum_{j=1}^{10} w_j x_{6j} \leq 110$
- For SectionID 7: $\sum_{j=1}^{10} w_j x_{7j} \leq 160$
- For SectionID 8: $\sum_{j=1}^{10} w_j x_{8j} \leq 140$

**Variable domains:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

---

**Where:**

- $v_j$ and $w_j$ for each product $j$ are:

| $j$ | $v_j$ | $w_j$ |
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

- $c_i$ for each section $i$ is:

| $i$ | $c_i$ |
|-----|-------|
| 1   | 100   |
| 2   | 150   |
| 3   | 120   |
| 4   | 130   |
| 5   | 90    |
| 6   | 110   |
| 7   | 160   |
| 8   | 140   |

---

**Summary:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{10} w_j x_{ij} \leq c_i \quad \forall i=1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,8;\ j=1,\ldots,10
\end{align*}
\]