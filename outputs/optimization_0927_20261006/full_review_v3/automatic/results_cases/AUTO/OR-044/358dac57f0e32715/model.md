Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$. All $x_{ij}$ are integer and nonnegative.

Indices:
- $i$ indexes SectionID $\in \{1,2,3,4,5,6,7,8\}$
- $j$ indexes ProductName $\in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters (from the data):

Section capacities:
\[
\begin{align*}
&\text{Section 1: } C_1 = 100 \\
&\text{Section 2: } C_2 = 150 \\
&\text{Section 3: } C_3 = 120 \\
&\text{Section 4: } C_4 = 130 \\
&\text{Section 5: } C_5 = 90 \\
&\text{Section 6: } C_6 = 110 \\
&\text{Section 7: } C_7 = 160 \\
&\text{Section 8: } C_8 = 140 \\
\end{align*}
\]

Product values and weights:
\[
\begin{array}{llll}
\text{ProductName} & \text{Value} & \text{Weight} \\
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

Model:

Objective:
\[
\max \sum_{i \in \{1,\ldots,8\}} \sum_{j \in \{1,\ldots,10\}} v_j \, x_{ij}
\]
where $v_j$ is the Value of product $j$.

Subject to, for each section $i$:
\[
\sum_{j=1}^{10} w_j \, x_{ij} \leq C_i \qquad \forall i \in \{1,\ldots,8\}
\]
where $w_j$ is the Weight of product $j$, and $C_i$ is the Capacity of section $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\; j \in \{1,\ldots,10\}
\]

Explicitly, with all coefficients:

\[
\begin{align*}
\max\quad & \sum_{i=1}^8 \Big( 10\,x_{i1} + 15\,x_{i2} + 8\,x_{i3} + 12\,x_{i4} + 20\,x_{i5} + 25\,x_{i6} + 5\,x_{i7} + 30\,x_{i8} + 18\,x_{i9} + 22\,x_{i10} \Big) \\
\text{s.t.}\quad
& 2x_{1,1} + 3x_{1,2} + 1x_{1,3} + 2x_{1,4} + 4x_{1,5} + 5x_{1,6} + 1x_{1,7} + 6x_{1,8} + 3x_{1,9} + 4x_{1,10} \leq 100 \\
& 2x_{2,1} + 3x_{2,2} + 1x_{2,3} + 2x_{2,4} + 4x_{2,5} + 5x_{2,6} + 1x_{2,7} + 6x_{2,8} + 3x_{2,9} + 4x_{2,10} \leq 150 \\
& 2x_{3,1} + 3x_{3,2} + 1x_{3,3} + 2x_{3,4} + 4x_{3,5} + 5x_{3,6} + 1x_{3,7} + 6x_{3,8} + 3x_{3,9} + 4x_{3,10} \leq 120 \\
& 2x_{4,1} + 3x_{4,2} + 1x_{4,3} + 2x_{4,4} + 4x_{4,5} + 5x_{4,6} + 1x_{4,7} + 6x_{4,8} + 3x_{4,9} + 4x_{4,10} \leq 130 \\
& 2x_{5,1} + 3x_{5,2} + 1x_{5,3} + 2x_{5,4} + 4x_{5,5} + 5x_{5,6} + 1x_{5,7} + 6x_{5,8} + 3x_{5,9} + 4x_{5,10} \leq 90 \\
& 2x_{6,1} + 3x_{6,2} + 1x_{6,3} + 2x_{6,4} + 4x_{6,5} + 5x_{6,6} + 1x_{6,7} + 6x_{6,8} + 3x_{6,9} + 4x_{6,10} \leq 110 \\
& 2x_{7,1} + 3x_{7,2} + 1x_{7,3} + 2x_{7,4} + 4x_{7,5} + 5x_{7,6} + 1x_{7,7} + 6x_{7,8} + 3x_{7,9} + 4x_{7,10} \leq 160 \\
& 2x_{8,1} + 3x_{8,2} + 1x_{8,3} + 2x_{8,4} + 4x_{8,5} + 5x_{8,6} + 1x_{8,7} + 6x_{8,8} + 3x_{8,9} + 4x_{8,10} \leq 140 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\; j \in \{1,\ldots,10\}
\end{align*}
\]