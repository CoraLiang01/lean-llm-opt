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

  | ProductName |
  |-------------|
  | 1           |
  | 2           |
  | 3           |
  | 4           |
  | 5           |
  | 6           |
  | 7           |
  | 8           |
  | 9           |
  | 10          |

  Product values and weights:

  - Product 1: $v_1 = 10$, $w_1 = 2$
  - Product 2: $v_2 = 15$, $w_2 = 3$
  - Product 3: $v_3 = 8$, $w_3 = 1$
  - Product 4: $v_4 = 12$, $w_4 = 2$
  - Product 5: $v_5 = 20$, $w_5 = 4$
  - Product 6: $v_6 = 25$, $w_6 = 5$
  - Product 7: $v_7 = 5$, $w_7 = 1$
  - Product 8: $v_8 = 30$, $w_8 = 6$
  - Product 9: $v_9 = 18$, $w_9 = 3$
  - Product 10: $v_{10} = 22$, $w_{10} = 4$

---

### Mathematical Model

**Decision Variables:**

$$
x_{ij} = \text{number of units of product } j \text{ to be placed in section } i, \quad x_{ij} \in \mathbb{Z}_{\geq 0}
$$

where $i \in \{1,2,3,4,5,6,7,8\}$ and $j \in \{1,2,3,4,5,6,7,8,9,10\}$.

---

**Objective:**

$$
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
$$

where $v_j$ is the value of product $j$ as listed above.

---

**Constraints:**

For each section $i$:

$$
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
$$

where $w_j$ is the weight (space requirement) of product $j$ and $C_i$ is the capacity of section $i$ as listed above.

For all $i, j$:

$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

**All data used:**

- SectionID and Capacity (in order):  
  1: 100, 2: 150, 3: 120, 4: 130, 5: 90, 6: 110, 7: 160, 8: 140

- ProductName, Value, Weight (in order):  
  1: 10, 2; 2: 15, 3; 3: 8, 1; 4: 12, 2; 5: 20, 4; 6: 25, 5; 7: 5, 1; 8: 30, 6; 9: 18, 3; 10: 22, 4

---

**Summary:**

Maximize total revenue from all sections, subject to each section's display space limit, by choosing integer numbers of each product for each section.