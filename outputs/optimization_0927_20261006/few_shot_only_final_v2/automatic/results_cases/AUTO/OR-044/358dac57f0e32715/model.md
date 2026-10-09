Let  
- $x_{ij}$ = number of units of product $j$ (ProductName $j$ from products.csv) to be placed in section $i$ (SectionID $i$ from capacity.csv), for all $i \in \{1,2,3,4,5,6,7,8\}$ and $j \in \{1,2,3,4,5,6,7,8,9,10\}$.

Parameters:  
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (space requirement) of product $j$ (from products.csv)
- $C_i$ = Capacity of section $i$ (from capacity.csv)

Data:  
From capacity.csv:  
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

From products.csv:  
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

Model:

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

**Subject to:**

For each section $i$ (SectionID $i$ from 1 to 8):  
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

**Variable domains:**  
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,3,4,5,6,7,8\},\ j \in \{1,2,3,4,5,6,7,8,9,10\}
$$

**Where:**

- $v_j$ and $w_j$ are as given in products.csv for ProductName $j$.
- $C_i$ is as given in capacity.csv for SectionID $i$.

**Explicitly:**

Letting $v_j$ and $w_j$ be:

- $v_1=10$, $w_1=2$
- $v_2=15$, $w_2=3$
- $v_3=8$, $w_3=1$
- $v_4=12$, $w_4=2$
- $v_5=20$, $w_5=4$
- $v_6=25$, $w_6=5$
- $v_7=5$, $w_7=1$
- $v_8=30$, $w_8=6$
- $v_9=18$, $w_9=3$
- $v_{10}=22$, $w_{10}=4$

And $C_i$ as above for each SectionID.

**Summary:**  
Maximize
$$
\sum_{i=1}^{8} \left(10x_{i1} + 15x_{i2} + 8x_{i3} + 12x_{i4} + 20x_{i5} + 25x_{i6} + 5x_{i7} + 30x_{i8} + 18x_{i9} + 22x_{i10}\right)
$$

Subject to, for each $i$:
$$
2x_{i1} + 3x_{i2} + 1x_{i3} + 2x_{i4} + 4x_{i5} + 5x_{i6} + 1x_{i7} + 6x_{i8} + 3x_{i9} + 4x_{i10} \leq C_i
$$
with $C_1=100$, $C_2=150$, $C_3=120$, $C_4=130$, $C_5=90$, $C_6=110$, $C_7=160$, $C_8=140$.

And
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,8;\ j=1,\ldots,10
$$