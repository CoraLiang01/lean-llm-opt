**Mathematical Model**

Let  
- $x_{ij}$ = number of units of product $j$ to be placed in section $i$ (integer, $\geq 0$)  
- $i$ indexes SectionID $\in \{1,2,3,4,5,6,7,8\}$  
- $j$ indexes ProductName $\in \{1,2,3,4,5,6,7,8,9,10\}$

Let  
- $v_j$ = Value of product $j$  
- $w_j$ = Weight (space requirement) of product $j$  
- $C_i$ = Capacity of section $i$

**Parameters (from data):**

- Section capacities:

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

- Product values and weights:

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

---

### Objective

Maximize total revenue:
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$
where $v_j$ is as above.

Explicitly:
\[
\max \Bigg[
\sum_{i=1}^{8} \Big(
10x_{i1} + 15x_{i2} + 8x_{i3} + 12x_{i4} + 20x_{i5} + 25x_{i6} + 5x_{i7} + 30x_{i8} + 18x_{i9} + 22x_{i10}
\Big)
\Bigg]
\]

---

### Constraints

#### 1. Section capacity constraints

For each section $i$:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i
\]
where $w_j$ and $C_i$ are as above.

Explicitly, for each section:

- Section 1:
  \[
  2x_{1,1} + 3x_{1,2} + 1x_{1,3} + 2x_{1,4} + 4x_{1,5} + 5x_{1,6} + 1x_{1,7} + 6x_{1,8} + 3x_{1,9} + 4x_{1,10} \leq 100
  \]
- Section 2:
  \[
  2x_{2,1} + 3x_{2,2} + 1x_{2,3} + 2x_{2,4} + 4x_{2,5} + 5x_{2,6} + 1x_{2,7} + 6x_{2,8} + 3x_{2,9} + 4x_{2,10} \leq 150
  \]
- Section 3:
  \[
  2x_{3,1} + 3x_{3,2} + 1x_{3,3} + 2x_{3,4} + 4x_{3,5} + 5x_{3,6} + 1x_{3,7} + 6x_{3,8} + 3x_{3,9} + 4x_{3,10} \leq 120
  \]
- Section 4:
  \[
  2x_{4,1} + 3x_{4,2} + 1x_{4,3} + 2x_{4,4} + 4x_{4,5} + 5x_{4,6} + 1x_{4,7} + 6x_{4,8} + 3x_{4,9} + 4x_{4,10} \leq 130
  \]
- Section 5:
  \[
  2x_{5,1} + 3x_{5,2} + 1x_{5,3} + 2x_{5,4} + 4x_{5,5} + 5x_{5,6} + 1x_{5,7} + 6x_{5,8} + 3x_{5,9} + 4x_{5,10} \leq 90
  \]
- Section 6:
  \[
  2x_{6,1} + 3x_{6,2} + 1x_{6,3} + 2x_{6,4} + 4x_{6,5} + 5x_{6,6} + 1x_{6,7} + 6x_{6,8} + 3x_{6,9} + 4x_{6,10} \leq 110
  \]
- Section 7:
  \[
  2x_{7,1} + 3x_{7,2} + 1x_{7,3} + 2x_{7,4} + 4x_{7,5} + 5x_{7,6} + 1x_{7,7} + 6x_{7,8} + 3x_{7,9} + 4x_{7,10} \leq 160
  \]
- Section 8:
  \[
  2x_{8,1} + 3x_{8,2} + 1x_{8,3} + 2x_{8,4} + 4x_{8,5} + 5x_{8,6} + 1x_{8,7} + 6x_{8,8} + 3x_{8,9} + 4x_{8,10} \leq 140
  \]

#### 2. Nonnegativity and integrality

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

---

**Summary of Model:**

Maximize
\[
\sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
\]
subject to, for each $i=1,\ldots,8$,
\[
\sum_{j=1}^{10} w_j x_{ij} \leq C_i
\]
and
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]