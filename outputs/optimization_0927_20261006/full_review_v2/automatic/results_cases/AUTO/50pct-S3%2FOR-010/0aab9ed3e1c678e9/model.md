Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $\geq 0$.

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

  Section capacities:

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

**Mathematical Model:**

Maximize total revenue:
$$
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \, x_{ij}
$$
where $v_j$ is the Value of product $j$.

Subject to, for each section $i$ (SectionID as above):

$$
\sum_{j=1}^{10} w_j \, x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

where $w_j$ is the Weight (space requirement) of product $j$, and $C_i$ is the Capacity of section $i$.

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\; j \in \{1,\ldots,10\}
$$

**Parameter values:**

- $v_j$ (Value): $[10, 15, 8, 12, 20, 25, 5, 30, 18, 22]$ for $j=1$ to $10$ (in source order).
- $w_j$ (Weight): $[2, 3, 1, 2, 4, 5, 1, 6, 3, 4]$ for $j=1$ to $10$ (in source order).
- $C_i$ (Capacity): $[100, 150, 120, 130, 90, 110, 160, 140]$ for $i=1$ to $8$ (in source order).

**Complete Formulation:**

$$
\begin{align*}
\max\quad & \sum_{i=1}^{8} \sum_{j=1}^{10} v_j\, x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{10} w_j\, x_{ij} \leq C_i \qquad \forall i=1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i=1,\ldots,8;\; j=1,\ldots,10
\end{align*}
$$

Where:

- $v_j$ and $w_j$ as above for each product $j$ (ProductName $1$ to $10$ in source order).
- $C_i$ as above for each section $i$ (SectionID $1$ to $8$ in source order).