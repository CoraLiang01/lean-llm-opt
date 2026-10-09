Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

**Indices:**
- $i$ indexes sections, with SectionID from the capacity.csv file.
- $j$ indexes products, with ProductName from the products.csv file.

**Parameters:**
- $c_i$: Capacity of section $i$ (from the "Capacity" column in capacity.csv).
- $v_j$: Value (price) of product $j$ (from the "Value" column in products.csv).
- $w_j$: Space requirement (weight) of product $j$ (from the "Weight" column in products.csv).

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

\[
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each section $i$ (SectionID as below):

\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Where:

- $c_1 = 100$, $c_2 = 150$, $c_3 = 120$, $c_4 = 130$, $c_5 = 90$, $c_6 = 110$, $c_7 = 160$, $c_8 = 140$
- $w_1 = 2$, $w_2 = 3$, $w_3 = 1$, $w_4 = 2$, $w_5 = 4$, $w_6 = 5$, $w_7 = 1$, $w_8 = 6$, $w_9 = 3$, $w_{10} = 4$

**Variable domains:**

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

---

**Explicitly:**

\[
\begin{align*}
\max\ & \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{10} w_j x_{1j} \leq 100 \\
& \sum_{j=1}^{10} w_j x_{2j} \leq 150 \\
& \sum_{j=1}^{10} w_j x_{3j} \leq 120 \\
& \sum_{j=1}^{10} w_j x_{4j} \leq 130 \\
& \sum_{j=1}^{10} w_j x_{5j} \leq 90 \\
& \sum_{j=1}^{10} w_j x_{6j} \leq 110 \\
& \sum_{j=1}^{10} w_j x_{7j} \leq 160 \\
& \sum_{j=1}^{10} w_j x_{8j} \leq 140 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,8;\ j=1,\ldots,10
\end{align*}
\]

Where for each $j$:

- $v_1=10$, $v_2=15$, $v_3=8$, $v_4=12$, $v_5=20$, $v_6=25$, $v_7=5$, $v_8=30$, $v_9=18$, $v_{10}=22$
- $w_1=2$, $w_2=3$, $w_3=1$, $w_4=2$, $w_5=4$, $w_6=5$, $w_7=1$, $w_8=6$, $w_9=3$, $w_{10}=4$

All data and identifiers are preserved in source order.