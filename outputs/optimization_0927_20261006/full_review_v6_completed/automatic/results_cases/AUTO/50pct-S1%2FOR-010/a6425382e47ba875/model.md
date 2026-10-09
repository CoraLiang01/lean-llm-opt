Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and nonnegative.

**Sets and Parameters:**

- Sections $i \in \{1,2,3,4,5,6,7,8\}$, with capacities $C_i$:
  - $C_1 = 100$
  - $C_2 = 150$
  - $C_3 = 120$
  - $C_4 = 130$
  - $C_5 = 90$
  - $C_6 = 110$
  - $C_7 = 160$
  - $C_8 = 140$

- Products $j \in \{1,2,3,4,5,6,7,8,9,10\}$, with values $v_j$ and weights $w_j$:
  - $v_1 = 10$, $w_1 = 2$
  - $v_2 = 15$, $w_2 = 3$
  - $v_3 = 8$, $w_3 = 1$
  - $v_4 = 12$, $w_4 = 2$
  - $v_5 = 20$, $w_5 = 4$
  - $v_6 = 25$, $w_6 = 5$
  - $v_7 = 5$, $w_7 = 1$
  - $v_8 = 30$, $w_8 = 6$
  - $v_9 = 18$, $w_9 = 3$
  - $v_{10} = 22$, $w_{10} = 4$

**Decision Variables:**

- $x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in \{1,\ldots,8\}$, $j \in \{1,\ldots,10\}$

**Objective:**

\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
\]

**Constraints:**

For each section $i$:
\[
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Numerical Data:**

- Section capacities:
  - Section 1: $C_1 = 100$
  - Section 2: $C_2 = 150$
  - Section 3: $C_3 = 120$
  - Section 4: $C_4 = 130$
  - Section 5: $C_5 = 90$
  - Section 6: $C_6 = 110$
  - Section 7: $C_7 = 160$
  - Section 8: $C_8 = 140$

- Product values and weights:
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