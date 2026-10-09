Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i$ indexes sections, with SectionID from the table below.
- $j$ indexes products, with ProductName from the table below.

**Parameters:**

- $c_i$: Capacity of section $i$ (from "Capacity" column).
- $v_j$: Value (price) of product $j$ (from "Value" column).
- $w_j$: Space requirement of product $j$ (from "Weight" column).

---

### Objective

$$
\max \sum_{i} \sum_{j} v_j \cdot x_{ij}
$$

---

### Constraints

For each section $i$ (SectionID):

$$
\sum_{j} w_j \cdot x_{ij} \leq c_i
$$

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

### Data Tables

#### Sections (from capacity.csv, in source order):

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

#### Products (from products.csv, in source order):

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

### Complete Model

$$
\begin{align*}
\max \quad & \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \cdot x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i, \quad \forall i \in \{1,\ldots,8\} \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\end{align*}
$$

Where $c_i$, $v_j$, and $w_j$ are as given in the tables above.