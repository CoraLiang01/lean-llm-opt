Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are required to be nonnegative integers.

**Indices:**
- $i$ indexes sections, with SectionID from the capacity.csv file.
- $j$ indexes products, with ProductName from the products.csv file.

**Parameters:**
- $c_i$: Capacity of section $i$ (from Capacity column in capacity.csv).
- $v_j$: Value (price) of product $j$ (from Value column in products.csv).
- $w_j$: Shelf space requirement (Weight) of product $j$ (from Weight column in products.csv).

---

### Objective

Maximize total revenue:
$$
\max \sum_{i \in \{\text{1,2,3,4,5,6,7,8}\}} \sum_{j \in \{\text{1,2,3,4,5,6,7,8,9,10}\}} v_j \cdot x_{ij}
$$

---

### Constraints

**Section capacity constraints:** For each section $i$,
$$
\sum_{j \in \{\text{1,2,3,4,5,6,7,8,9,10}\}} w_j \cdot x_{ij} \leq c_i
\qquad \forall i \in \{\text{1,2,3,4,5,6,7,8}\}
$$

**Integrality and nonnegativity:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
\qquad \forall i \in \{\text{1,2,3,4,5,6,7,8}\},\ 
         j \in \{\text{1,2,3,4,5,6,7,8,9,10}\}
$$

---

### Parameter Tables

**Sections (from capacity.csv):**

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

**Products (from products.csv):**

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

**Complete Model:**

Maximize
$$
\sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
$$

Subject to, for each $i=1,\ldots,8$:
$$
\sum_{j=1}^{10} w_j x_{ij} \leq c_i
$$

and
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,8;\ j=1,\ldots,10
$$

where the parameters $c_i$, $v_j$, and $w_j$ are as given in the tables above.