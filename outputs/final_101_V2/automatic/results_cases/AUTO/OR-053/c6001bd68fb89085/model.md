Let $x_{ij}$ be the number of units of product $j$ (ProductName $j$) to be placed on shelf $i$ (ShelfID $i$), for $i = 1,\ldots,10$ and $j = 1,\ldots,20$.

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the value of product $j$ as given below.

Subject to, for each shelf $i$:
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]
where $w_j$ is the weight of product $j$ and $C_i$ is the capacity of shelf $i$ as given below.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\]

Parameters (from the retrieved data):

Shelf capacities:
\[
\begin{array}{ll}
C_1 = 500 & C_6 = 900 \\
C_2 = 700 & C_7 = 650 \\
C_3 = 600 & C_8 = 750 \\
C_4 = 800 & C_9 = 820 \\
C_5 = 550 & C_{10} = 570 \\
\end{array}
\]

Product values and weights:
\[
\begin{array}{lll}
j & v_j & w_j \\
1 & 50 & 10 \\
2 & 70 & 20 \\
3 & 30 & 5 \\
4 & 60 & 15 \\
5 & 80 & 25 \\
6 & 90 & 30 \\
7 & 40 & 12 \\
8 & 100 & 35 \\
9 & 55 & 10 \\
10 & 75 & 20 \\
11 & 65 & 18 \\
12 & 95 & 28 \\
13 & 45 & 8 \\
14 & 85 & 22 \\
15 & 70 & 25 \\
16 & 110 & 40 \\
17 & 50 & 14 \\
18 & 60 & 16 \\
19 & 120 & 50 \\
20 & 100 & 30 \\
\end{array}
\]

Full model:
\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\ & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\end{align*}
\]
with $v_j$, $w_j$, and $C_i$ as listed above.