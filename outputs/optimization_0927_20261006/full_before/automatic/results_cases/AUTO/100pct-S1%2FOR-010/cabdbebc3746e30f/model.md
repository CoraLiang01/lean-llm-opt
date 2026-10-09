Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let:
- $i$ index SectionID from capacity.csv: $i \in \{1,2,3,4,5,6,7,8\}$
- $j$ index ProductName from products.csv: $j \in \{1,2,3,4,5,6,7,8,9,10\}$
- $c_i$ = Capacity of section $i$
- $v_j$ = Value of product $j$
- $w_j$ = Weight (space requirement) of product $j$

Parameters from the data:

Section capacities:
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

Product values and weights:
\[
\begin{align*}
&\text{ProductName} \quad v_j \quad w_j \\
1 &\quad 10 \quad 2 \\
2 &\quad 15 \quad 3 \\
3 &\quad 8 \quad 1 \\
4 &\quad 12 \quad 2 \\
5 &\quad 20 \quad 4 \\
6 &\quad 25 \quad 5 \\
7 &\quad 5 \quad 1 \\
8 &\quad 30 \quad 6 \\
9 &\quad 18 \quad 3 \\
10 &\quad 22 \quad 4 \\
\end{align*}
\]

Objective:
\[
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j x_{ij}
\]

Subject to, for each section $i$:
\[
\sum_{j=1}^{10} w_j x_{ij} \leq c_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

Where:
- $v_j$ and $w_j$ are as listed above for each ProductName $j$
- $c_i$ is as listed above for each SectionID $i$