Let $x_{ij}$ be the number of units of product $j$ to be placed in section $i$.

Indices:
- $i$: SectionID $\in \{1,2,3,4,5,6,7,8\}$
- $j$: ProductName $\in \{1,2,3,4,5,6,7,8,9,10\}$

Parameters:
- $v_j$: Value of product $j$ (from Value column)
- $w_j$: Weight (space requirement) of product $j$ (from Weight column)
- $C_i$: Capacity of section $i$ (from Capacity column)

Data:

Section Capacities:
\[
\begin{array}{ll}
\text{SectionID} & \text{Capacity} \\
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

Product Data:
\[
\begin{array}{lll}
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
\max \sum_{i=1}^{8} \sum_{j=1}^{10} v_j \cdot x_{ij}
\]

Subject to:

Section capacity constraints:
\[
\sum_{j=1}^{10} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in \{1,2,3,4,5,6,7,8\}
\]

Integrality and nonnegativity:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,\ldots,8\},\ j \in \{1,\ldots,10\}
\]

Where:
\[
\begin{align*}
v_1 &= 10,\quad w_1 = 2 \\
v_2 &= 15,\quad w_2 = 3 \\
v_3 &= 8,\quad w_3 = 1 \\
v_4 &= 12,\quad w_4 = 2 \\
v_5 &= 20,\quad w_5 = 4 \\
v_6 &= 25,\quad w_6 = 5 \\
v_7 &= 5,\quad w_7 = 1 \\
v_8 &= 30,\quad w_8 = 6 \\
v_9 &= 18,\quad w_9 = 3 \\
v_{10} &= 22,\quad w_{10} = 4 \\
\end{align*}
\]
and
\[
\begin{align*}
C_1 &= 100 \\
C_2 &= 150 \\
C_3 &= 120 \\
C_4 &= 130 \\
C_5 &= 90 \\
C_6 &= 110 \\
C_7 &= 160 \\
C_8 &= 140 \\
\end{align*}
\]

All variables $x_{ij}$ are nonnegative integers.