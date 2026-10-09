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

**Mathematical Model:**

Objective:
\[
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$ as above.

Subject to, for each section $i$:

\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
\]
where $w_j$ is the Weight (space requirement) of product $j$ and $C_i$ is the Capacity of section $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

**Explicit Data Used:**

- Section capacities (by SectionID, in order): 100, 150, 120, 130, 90, 110, 160, 140
- Product values and weights (by ProductName, in order):  
  - 1: Value 10, Weight 2  
  - 2: Value 15, Weight 3  
  - 3: Value 8, Weight 1  
  - 4: Value 12, Weight 2  
  - 5: Value 20, Weight 4  
  - 6: Value 25, Weight 5  
  - 7: Value 5, Weight 1  
  - 8: Value 30, Weight 6  
  - 9: Value 18, Weight 3  
  - 10: Value 22, Weight 4

**Summary:**

\[
\begin{align*}
\max\ & \sum_{i=1}^8 \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{10} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,8;\ j=1,\ldots,10
\end{align*}
\]
where $v_j$ and $w_j$ are as listed above, and $C_i$ are the section capacities.