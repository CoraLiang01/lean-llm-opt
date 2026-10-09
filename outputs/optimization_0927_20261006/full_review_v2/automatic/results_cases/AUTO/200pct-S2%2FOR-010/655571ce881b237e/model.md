Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Sections (from capacity.csv, in source order):

  | SectionID |
  |-----------|
  |     1     |
  |     2     |
  |     3     |
  |     4     |
  |     5     |
  |     6     |
  |     7     |
  |     8     |

  Section capacities:

  - $C_1 = 100$
  - $C_2 = 150$
  - $C_3 = 120$
  - $C_4 = 130$
  - $C_5 = 90$
  - $C_6 = 110$
  - $C_7 = 160$
  - $C_8 = 140$

- Products (from products.csv, in source order):

  | ProductName |
  |-------------|
  |     1       |
  |     2       |
  |     3       |
  |     4       |
  |     5       |
  |     6       |
  |     7       |
  |     8       |
  |     9       |
  |    10       |

  Product values and space requirements:

  - Product 1: Value = 10, Weight = 2
  - Product 2: Value = 15, Weight = 3
  - Product 3: Value = 8,  Weight = 1
  - Product 4: Value = 12, Weight = 2
  - Product 5: Value = 20, Weight = 4
  - Product 6: Value = 25, Weight = 5
  - Product 7: Value = 5,  Weight = 1
  - Product 8: Value = 30, Weight = 6
  - Product 9: Value = 18, Weight = 3
  - Product 10: Value = 22, Weight = 4

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**

$$
\max \sum_{i=1}^8 \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

where $v_j$ is the Value of product $j$ as listed above.

**Constraints:**

For each section $i$ (SectionID as above):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

where $w_j$ is the Weight (space requirement) of product $j$ and $C_i$ is the Capacity of section $i$.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**Explicit Coefficients:**

- $v_1 = 10$, $w_1 = 2$
- $v_2 = 15$, $w_2 = 3$
- $v_3 = 8$,  $w_3 = 1$
- $v_4 = 12$, $w_4 = 2$
- $v_5 = 20$, $w_5 = 4$
- $v_6 = 25$, $w_6 = 5$
- $v_7 = 5$,  $w_7 = 1$
- $v_8 = 30$, $w_8 = 6$
- $v_9 = 18$, $w_9 = 3$
- $v_{10} = 22$, $w_{10} = 4$

- $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^8 \sum_{j=1}^{10} v_j x_{ij} \\
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
$$

where $v_j$ and $w_j$ are as listed above.