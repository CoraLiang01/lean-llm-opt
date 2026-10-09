Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

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

  Section capacities:

  - $C_1 = 100$
  - $C_2 = 150$
  - $C_3 = 120$
  - $C_4 = 130$
  - $C_5 = 90$
  - $C_6 = 110$
  - $C_7 = 160$
  - $C_8 = 140$

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

Let $V_j$ be the Value of product $j$, and $W_j$ be the Weight (space requirement) of product $j$.

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,2,3,4,5,6,7,8\},\ j \in \{1,2,3,4,5,6,7,8,9,10\}
$$

**Objective:**

$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} V_j \cdot x_{ij}
$$

where

- $V_1 = 10$, $V_2 = 15$, $V_3 = 8$, $V_4 = 12$, $V_5 = 20$, $V_6 = 25$, $V_7 = 5$, $V_8 = 30$, $V_9 = 18$, $V_{10} = 22$

**Constraints:**

For each section $i$:

$$
\sum_{j=1}^{10} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

where

- $W_1 = 2$, $W_2 = 3$, $W_3 = 1$, $W_4 = 2$, $W_5 = 4$, $W_6 = 5$, $W_7 = 1$, $W_8 = 6$, $W_9 = 3$, $W_{10} = 4$
- $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

---

**Complete Model (with explicit coefficients):**

$$
\max \sum_{i=1}^{8} \left(
10\,x_{i1} + 15\,x_{i2} + 8\,x_{i3} + 12\,x_{i4} + 20\,x_{i5} + 25\,x_{i6} + 5\,x_{i7} + 30\,x_{i8} + 18\,x_{i9} + 22\,x_{i10}
\right)
$$

Subject to, for each $i$:

$$
2\,x_{i1} + 3\,x_{i2} + 1\,x_{i3} + 2\,x_{i4} + 4\,x_{i5} + 5\,x_{i6} + 1\,x_{i7} + 6\,x_{i8} + 3\,x_{i9} + 4\,x_{i10} \leq C_i
$$

with $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$.

and

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,8;\ j=1,\ldots,10
$$