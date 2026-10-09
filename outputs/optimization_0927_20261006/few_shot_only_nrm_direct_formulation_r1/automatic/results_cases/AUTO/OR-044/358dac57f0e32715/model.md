**Sets and Indices:**
- Let $i$ index sections, $i \in \{1,2,3,4,5,6,7,8\}$ (from "SectionID" in capacity.csv)
- Let $j$ index products, $j \in \{1,2,3,4,5,6,7,8,9,10\}$ (from "ProductName" in products.csv)

**Parameters:**
- $c_i$ = capacity of section $i$ (from "Capacity" in capacity.csv)
- $v_j$ = value of product $j$ (from "Value" in products.csv)
- $w_j$ = space requirement of product $j$ (from "Weight" in products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed in section $i$

**Objective:**
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]

**Constraints:**

For each section $i$:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Parameter Values (from CSVs):**

From capacity.csv:
\[
\begin{align*}
c_1 &= 100 \\
c_2 &= 150 \\
c_3 &= 120 \\
c_4 &= 130 \\
c_5 &= 90 \\
c_6 &= 110 \\
c_7 &= 160 \\
c_8 &= 140 \\
\end{align*}
\]

From products.csv:
\[
\begin{array}{cccc}
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

**Full Model:**

\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]

Subject to, for each $i$:
\[
2x_{i1} + 3x_{i2} + 1x_{i3} + 2x_{i4} + 4x_{i5} + 5x_{i6} + 1x_{i7} + 6x_{i8} + 3x_{i9} + 4x_{i10} \leq c_i
\]
where $c_i$ is as above for each $i=1,\ldots,8$.

And for all $i=1,\ldots,8$, $j=1,\ldots,10$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]