Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $\geq 0$.

**Parameters:**

- Sections (from capacity.csv, in order):

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

**Model:**

Maximize total revenue:
$$
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \cdot x_{ij}
$$
where $v_j$ is the Value of product $j$.

Subject to, for each section $i$ (SectionID as above):

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of section $i$.

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Parameter values (in source order):**

- $C_1=100$, $C_2=150$, $C_3=120$, $C_4=130$, $C_5=90$, $C_6=110$, $C_7=160$, $C_8=140$
- $v_1=10$, $v_2=15$, $v_3=8$, $v_4=12$, $v_5=20$, $v_6=25$, $v_7=5$, $v_8=30$, $v_9=18$, $v_{10}=22$
- $w_1=2$, $w_2=3$, $w_3=1$, $w_4=2$, $w_5=4$, $w_6=5$, $w_7=1$, $w_8=6$, $w_9=3$, $w_{10}=4$

**Decision variables:**

- $x_{ij}$: integer, $\geq 0$, for $i=1,\ldots,8$ (SectionID), $j=1,\ldots,10$ (ProductName)

**Complete Formulation:**

$$
\begin{align*}
\max\ & \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i=1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,8,\ j=1,\ldots,10
\end{align*}
$$

with all parameter values as above, and all indices and coefficients preserved in source order.