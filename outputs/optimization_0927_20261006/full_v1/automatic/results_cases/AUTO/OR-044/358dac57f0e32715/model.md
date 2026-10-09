Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and $x_{ij} \geq 0$.

Let $S$ be the set of sections (indexed by SectionID), and $P$ be the set of products (indexed by ProductName).

Define:
- $v_j$: Value of product $j$
- $w_j$: Weight (space requirement) of product $j$
- $C_i$: Capacity of section $i$

From the data:

Sections and capacities:
- Section 1: $C_1 = 100$
- Section 2: $C_2 = 150$
- Section 3: $C_3 = 120$
- Section 4: $C_4 = 130$
- Section 5: $C_5 = 90$
- Section 6: $C_6 = 110$
- Section 7: $C_7 = 160$
- Section 8: $C_8 = 140$

Products, values, and weights:
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

The mathematical model is:

Objective:
$$
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j x_{ij}
$$

Subject to, for each section $i$:
$$
\sum_{j=1}^{10} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
$$

Variable domains:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
$$

Where:

\[
\begin{align*}
& C_1 = 100,\quad C_2 = 150,\quad C_3 = 120,\quad C_4 = 130, \\
& C_5 = 90,\quad C_6 = 110,\quad C_7 = 160,\quad C_8 = 140 \\
& (v_1, w_1) = (10, 2),\quad (v_2, w_2) = (15, 3),\quad (v_3, w_3) = (8, 1), \\
& (v_4, w_4) = (12, 2),\quad (v_5, w_5) = (20, 4),\quad (v_6, w_6) = (25, 5), \\
& (v_7, w_7) = (5, 1),\quad (v_8, w_8) = (30, 6),\quad (v_9, w_9) = (18, 3),\quad (v_{10}, w_{10}) = (22, 4)
\end{align*}
\]

All $x_{ij}$ are integer and nonnegative.