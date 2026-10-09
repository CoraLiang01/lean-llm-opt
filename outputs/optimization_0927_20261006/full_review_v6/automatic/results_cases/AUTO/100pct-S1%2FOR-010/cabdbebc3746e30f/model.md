Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are required to be nonnegative integers.

**Indices:**
- $i \in \{1,2,3,4,5,6,7,8\}$ (SectionID from capacity.csv)
- $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of section $i$ (from "Capacity" in capacity.csv)
- $v_j$ = Value (price) of product $j$ (from "Value" in products.csv)
- $w_j$ = Weight (space requirement) of product $j$ (from "Weight" in products.csv)

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

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$
where $v_j$ is as above.

**Section Capacity Constraints:**
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,8\}
$$
where $w_j$ and $c_i$ are as above.

**Variable Domains:**
$$
x_{ij} \in \{0,1,2,\ldots\} \qquad \forall i,j
$$

---

**Explicit Data Table for Parameters:**

- $c_1 = 100$, $c_2 = 150$, $c_3 = 120$, $c_4 = 130$, $c_5 = 90$, $c_6 = 110$, $c_7 = 160$, $c_8 = 140$
- $(v_1, w_1) = (10, 2)$
- $(v_2, w_2) = (15, 3)$
- $(v_3, w_3) = (8, 1)$
- $(v_4, w_4) = (12, 2)$
- $(v_5, w_5) = (20, 4)$
- $(v_6, w_6) = (25, 5)$
- $(v_7, w_7) = (5, 1)$
- $(v_8, w_8) = (30, 6)$
- $(v_9, w_9) = (18, 3)$
- $(v_{10}, w_{10}) = (22, 4)$

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,8;\ j = 1,\ldots,10
\end{align*}
$$

where all parameters are as listed above, and all indices and coefficients are preserved in source order.