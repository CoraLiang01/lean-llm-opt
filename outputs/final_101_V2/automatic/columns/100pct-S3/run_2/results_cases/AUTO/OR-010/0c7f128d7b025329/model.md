Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

**Parameters:**

- Sections (from capacity.csv, SectionID): $i \in \{1,2,3,4,5,6,7,8\}$
- Products (from products.csv, ProductName): $j \in \{1,2,3,4,5,6,7,8,9,10\}$
- Capacity of section $i$: $C_i$ (from "Capacity" column)
- Value of product $j$: $v_j$ (from "Value" column)
- Space requirement of product $j$: $w_j$ (from "Weight" column)

**Data:**

Section capacities:
\[
\begin{array}{ll}
C_1 = 100 & C_5 = 90 \\
C_2 = 150 & C_6 = 110 \\
C_3 = 120 & C_7 = 160 \\
C_4 = 130 & C_8 = 140 \\
\end{array}
\]

Product values and weights:
\[
\begin{array}{lll}
j & v_j & w_j \\
1 & 10 & 2 \\
2 & 15 & 3 \\
3 & 8 & 1 \\
4 & 12 & 2 \\
5 & 20 & 4 \\
6 & 25 & 5 \\
7 & 5 & 1 \\
8 & 30 & 6 \\
9 & 18 & 3 \\
10 & 22 & 4 \\
\end{array}
\]

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
\]

**Subject to:**

Section capacity constraints (for each section $i$):
\[
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

---

**Explicitly:**

For each section $i$:

- Section 1: $\sum_{j=1}^{10} w_j x_{1j} \leq 100$
- Section 2: $\sum_{j=1}^{10} w_j x_{2j} \leq 150$
- Section 3: $\sum_{j=1}^{10} w_j x_{3j} \leq 120$
- Section 4: $\sum_{j=1}^{10} w_j x_{4j} \leq 130$
- Section 5: $\sum_{j=1}^{10} w_j x_{5j} \leq 90$
- Section 6: $\sum_{j=1}^{10} w_j x_{6j} \leq 110$
- Section 7: $\sum_{j=1}^{10} w_j x_{7j} \leq 160$
- Section 8: $\sum_{j=1}^{10} w_j x_{8j} \leq 140$

Where for all $j$:
- $w_1=2$, $w_2=3$, $w_3=1$, $w_4=2$, $w_5=4$, $w_6=5$, $w_7=1$, $w_8=6$, $w_9=3$, $w_{10}=4$
- $v_1=10$, $v_2=15$, $v_3=8$, $v_4=12$, $v_5=20$, $v_6=25$, $v_7=5$, $v_8=30$, $v_9=18$, $v_{10}=22$

**Decision variables:**
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \text{for all } i=1,\ldots,8;\ j=1,\ldots,10
\]