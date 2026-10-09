Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $\geq 0$.

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

- Products $j \in \{1,2,3,4,5,6,7,8,9,10\}$, with values $v_j$ and space requirements $w_j$:
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

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]

**Subject to:**

For each section $i$:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

---

**Data Used:**

- Section capacities (from capacity.csv):
  - SectionID: 1, Capacity: 100
  - SectionID: 2, Capacity: 150
  - SectionID: 3, Capacity: 120
  - SectionID: 4, Capacity: 130
  - SectionID: 5, Capacity: 90
  - SectionID: 6, Capacity: 110
  - SectionID: 7, Capacity: 160
  - SectionID: 8, Capacity: 140

- Product values and space requirements (from products.csv):
  - ProductName: 1, Value: 10, Weight: 2
  - ProductName: 2, Value: 15, Weight: 3
  - ProductName: 3, Value: 8, Weight: 1
  - ProductName: 4, Value: 12, Weight: 2
  - ProductName: 5, Value: 20, Weight: 4
  - ProductName: 6, Value: 25, Weight: 5
  - ProductName: 7, Value: 5, Weight: 1
  - ProductName: 8, Value: 30, Weight: 6
  - ProductName: 9, Value: 18, Weight: 3
  - ProductName: 10, Value: 22, Weight: 4

---

**Decision variables:**
- $x_{ij}$: integer, $\geq 0$, number of units of product $j$ in section $i$.

**Objective:**
- Maximize total revenue.

**Constraints:**
- For each section, total space used by all products does not exceed the section's capacity.