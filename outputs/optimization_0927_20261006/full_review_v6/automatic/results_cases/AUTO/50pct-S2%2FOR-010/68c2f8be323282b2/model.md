Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Parameters (from retrieved data, in source order):**

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

  Let $V_j$ be the Value (price) of product $j$, and $W_j$ be the Weight (space requirement) of product $j$.

---

**Mathematical Model:**

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**

Maximize total revenue:
$$
\max \sum_{i=1}^8 \sum_{j=1}^{10} V_j \cdot x_{ij}
$$

**Subject to:**

Section capacity constraints (for each section $i$):

$$
\sum_{j=1}^{10} W_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

**Variable domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**Parameter Table (for reference):**

| $i$ (SectionID) | $C_i$ |
|-----------------|-------|
| 1               | 100   |
| 2               | 150   |
| 3               | 120   |
| 4               | 130   |
| 5               | 90    |
| 6               | 110   |
| 7               | 160   |
| 8               | 140   |

| $j$ (ProductName) | $V_j$ | $W_j$ |
|-------------------|-------|-------|
| 1                 | 10    | 2     |
| 2                 | 15    | 3     |
| 3                 | 8     | 1     |
| 4                 | 12    | 2     |
| 5                 | 20    | 4     |
| 6                 | 25    | 5     |
| 7                 | 5     | 1     |
| 8                 | 30    | 6     |
| 9                 | 18    | 3     |
| 10                | 22    | 4     |

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^8 \sum_{j=1}^{10} V_j \cdot x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{10} W_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,8,\ j = 1,\ldots,10
\end{align*}
$$

Where $V_j$ and $W_j$ are as given above, and $C_i$ as per section.