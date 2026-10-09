Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are nonnegative integers.

Define:
- Sections $i \in \{1,2,3,4,5,6,7,8\}$, with capacities $C_i$ as below.
- Products $j \in \{1,2,3,4,5,6,7,8,9,10\}$, with values $v_j$ and space requirements $w_j$ as below.

#### Data

| SectionID ($i$) | Capacity ($C_i$) |
|:---------------:|:----------------:|
| 1               | 100              |
| 2               | 150              |
| 3               | 120              |
| 4               | 130              |
| 5               | 90               |
| 6               | 110              |
| 7               | 160              |
| 8               | 140              |

| ProductName ($j$) | Value ($v_j$) | Weight ($w_j$) |
|:-----------------:|:-------------:|:--------------:|
| 1                 | 10            | 2              |
| 2                 | 15            | 3              |
| 3                 | 8             | 1              |
| 4                 | 12            | 2              |
| 5                 | 20            | 4              |
| 6                 | 25            | 5              |
| 7                 | 5             | 1              |
| 8                 | 30            | 6              |
| 9                 | 18            | 3              |
| 10                | 22            | 4              |

#### Mathematical Model

**Decision Variables:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

**Objective:**
$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

**Section Capacity Constraints:**
$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

**Variable Domains:**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i,j
$$

**Where:**

- $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$
- $(v_1,\ldots,v_{10}) = (10, 15, 8, 12, 20, 25, 5, 30, 18, 22)$
- $(w_1,\ldots,w_{10}) = (2, 3, 1, 2, 4, 5, 1, 6, 3, 4)$

**Summary:**

- Maximize total revenue from all sections and products.
- For each section, the total space used by all products cannot exceed its capacity.
- All product allocations are nonnegative integers.