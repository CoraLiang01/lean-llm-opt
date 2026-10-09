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

  Let $v_j$ be the Value and $w_j$ be the Weight (space requirement) of product $j$.

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

For each section $i$:

$$
\sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,\ldots,8\}
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

and $c_i$ is as above for each SectionID.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^8 \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i = 1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,8,\ j = 1,\ldots,10
\end{align*}
$$

with $v_j$, $w_j$, and $c_i$ as specified above.