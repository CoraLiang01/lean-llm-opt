Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Sections (from capacity.csv, in order):

  | SectionID |
  |-----------|
  | 1         |
  | 2         |
  | 3         |
  | 4         |
  | 5         |
  | 6         |
  | 7         |
  | 8         |

  Each section $i$ has capacity $C_i$:

  $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$

- Products (from products.csv, in order):

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

Let $v_j$ be the Value of product $j$, and $w_j$ be the Weight (space requirement) of product $j$.

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**

$$
\max \sum_{i=1}^8 \sum_{j=1}^{10} v_j x_{ij}
$$

where $v_j$ is as follows:

- $v_1 = 10$
- $v_2 = 15$
- $v_3 = 8$
- $v_4 = 12$
- $v_5 = 20$
- $v_6 = 25$
- $v_7 = 5$
- $v_8 = 30$
- $v_9 = 18$
- $v_{10} = 22$

**Constraints:**

For each section $i$ (SectionID as above):

$$
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

where $w_j$ is as follows:

- $w_1 = 2$
- $w_2 = 3$
- $w_3 = 1$
- $w_4 = 2$
- $w_5 = 4$
- $w_6 = 5$
- $w_7 = 1$
- $w_8 = 6$
- $w_9 = 3$
- $w_{10} = 4$

and $C_i$ is as above for each SectionID.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**Summary of Model:**

- Maximize total revenue from all products and sections.
- For each section, the total space used by all products cannot exceed its capacity.
- All product allocations are nonnegative integers.
- All identifiers and coefficients are as retrieved and in original order.