Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID) and $j \in \{1,2,\ldots,20\}$ (ProductName).

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the value of product $j$.

Subject to:

For each shelf $i$ (ShelfID):

\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]
where $w_j$ is the weight of product $j$, and $C_i$ is the capacity of shelf $i$.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\]

Parameters (from the data):

Shelf capacities:
\[
\begin{align*}
C_1 &= 500 \\
C_2 &= 700 \\
C_3 &= 600 \\
C_4 &= 800 \\
C_5 &= 550 \\
C_6 &= 900 \\
C_7 &= 650 \\
C_8 &= 750 \\
C_9 &= 820 \\
C_{10} &= 570 \\
\end{align*}
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

Summary:

- Decision variables: $x_{ij}$ = number of units of product $j$ on shelf $i$, integer, $\geq 0$
- Objective: maximize total value across all shelves
- Constraints: for each shelf, total weight of products does not exceed shelf capacity

Complete Model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\end{align*}
\]
with $v_j$, $w_j$, $C_i$ as given above.