Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

**Parameters:**

- Sections (from capacity.csv):

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

  $$
  \begin{align*}
  c_1 &= 100 \\
  c_2 &= 150 \\
  c_3 &= 120 \\
  c_4 &= 130 \\
  c_5 &= 90 \\
  c_6 &= 110 \\
  c_7 &= 160 \\
  c_8 &= 140 \\
  \end{align*}
  $$

- Products (from products.csv):

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

where $v_j$ is as given above.

**Constraints:**

For each section $i$:

$$
\sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,8\}
$$

where $w_j$ and $c_i$ are as given above.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**All parameters and identifiers are as retrieved and used in the model.**