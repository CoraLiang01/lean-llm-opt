**Sets and Indices:**
- $i \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}\}$ (SectionID from capacity.csv)
- $j \in \{\text{1}, \text{2}, \text{3}, \text{4}, \text{5}, \text{6}, \text{7}, \text{8}, \text{9}, \text{10}\}$ (ProductName from products.csv)

**Parameters:**
- $c_i$ = Capacity of section $i$ (from capacity.csv)
- $v_j$ = Value of product $j$ (from products.csv)
- $w_j$ = Weight (space requirement) of product $j$ (from products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ to stock in section $i$

**Objective:**
\[
\max \sum_{i \in \{1,2,3,4,5,6,7,8\}} \sum_{j \in \{1,2,3,4,5,6,7,8,9,10\}} v_j \cdot x_{ij}
\]

**Subject to:**

For each section $i$:
\[
\sum_{j \in \{1,2,3,4,5,6,7,8,9,10\}} w_j \cdot x_{ij} \leq c_i
\quad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

For all $i, j$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

**Parameter Values (from CSVs):**

From capacity.csv:
\[
\begin{array}{lll}
\text{SectionID} & c_i \\
1 & 100 \\
2 & 150 \\
3 & 120 \\
4 & 130 \\
5 & 90 \\
6 & 110 \\
7 & 160 \\
8 & 140 \\
\end{array}
\]

From products.csv:
\[
\begin{array}{llll}
\text{ProductName} & v_j & w_j \\
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

Where $c_i$ is as above for each section $i$.

And for all $i=1,\ldots,8$, $j=1,\ldots,10$:
\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]