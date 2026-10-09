Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

**Parameters:**

- Sections (from capacity.csv, in source order):

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

  $$
  \begin{align*}
  C_1 &= 100 \\
  C_2 &= 150 \\
  C_3 &= 120 \\
  C_4 &= 130 \\
  C_5 &= 90 \\
  C_6 &= 110 \\
  C_7 &= 160 \\
  C_8 &= 140 \\
  \end{align*}
  $$

- Products (from products.csv, in source order):

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

Let $p_j$ be the Value of product $j$, and $w_j$ be the Weight (space requirement) of product $j$.

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**

$$
\max \sum_{i=1}^8 \sum_{j=1}^{10} p_j x_{ij}
$$

where $p_j$ is as follows:

\[
\begin{align*}
p_1 &= 10 \\
p_2 &= 15 \\
p_3 &= 8 \\
p_4 &= 12 \\
p_5 &= 20 \\
p_6 &= 25 \\
p_7 &= 5 \\
p_8 &= 30 \\
p_9 &= 18 \\
p_{10} &= 22 \\
\end{align*}
\]

**Constraints:**

For each section $i$ (SectionID as above):

$$
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

where $w_j$ is as follows:

\[
\begin{align*}
w_1 &= 2 \\
w_2 &= 3 \\
w_3 &= 1 \\
w_4 &= 2 \\
w_5 &= 4 \\
w_6 &= 5 \\
w_7 &= 1 \\
w_8 &= 6 \\
w_9 &= 3 \\
w_{10} &= 4 \\
\end{align*}
\]

And $C_i$ as above.

**Variable Domains:**

$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

---

**Complete Model:**

$$
\begin{align*}
\max\ & \sum_{i=1}^8 \sum_{j=1}^{10} p_j x_{ij} \\
\text{s.t.} \quad & \sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,8,\ j = 1,\ldots,10 \\
\end{align*}
$$

where $p_j$, $w_j$, and $C_i$ are as specified above, and $x_{ij}$ is the number of units of product $j$ to be placed in section $i$.