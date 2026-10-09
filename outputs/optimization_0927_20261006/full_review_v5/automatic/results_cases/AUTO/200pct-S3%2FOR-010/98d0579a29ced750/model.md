Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

**Indices:**
- $i$ indexes sections, with SectionID from the table below.
- $j$ indexes products, with ProductName from the table below.

---

### Objective Function

\[
\max \sum_{i \in \{\text{SectionID}\}} \sum_{j \in \{\text{ProductName}\}} v_j \cdot x_{ij}
\]

where $v_j$ is the Value of product $j$.

---

### Constraints

#### 1. Section Capacity Constraints

For each section $i$ (SectionID), the total space used by all products cannot exceed its Capacity:

\[
\sum_{j} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{\text{SectionID}\}
\]

where $w_j$ is the Weight (shelf space requirement) of product $j$, and $C_i$ is the Capacity of section $i$.

#### 2. Nonnegativity and Integrality

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i, j
\]

---

### Parameter Tables (from retrieved data)

#### Sections (capacity.csv, in source order)

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

#### Products (products.csv, in source order)

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

**Summary of Model:**

- Decision variables: $x_{ij}$ = number of units of product $j$ in section $i$, integer, $\geq 0$
- Objective: maximize total revenue $\sum_{i,j} v_j x_{ij}$
- For each section $i$: $\sum_j w_j x_{ij} \leq C_i$ (do not exceed section capacity)
- All parameters as above, using explicit SectionID and ProductName as indices.