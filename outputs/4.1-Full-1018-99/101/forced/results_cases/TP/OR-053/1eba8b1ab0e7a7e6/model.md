Let $I = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves (with ShelfID as given), and $J = \{1,2,\ldots,20\}$ be the set of products (with ProductName as given).

Let $x_{ij} \in \mathbb{Z}_{\geq 0}$ denote the number of units of product $j$ placed on shelf $i$.

Parameters:

- Shelf capacities $C_i$:

\[
\begin{array}{ll}
C_1 = 500 & C_6 = 900 \\
C_2 = 700 & C_7 = 650 \\
C_3 = 600 & C_8 = 750 \\
C_4 = 800 & C_9 = 820 \\
C_5 = 550 & C_{10} = 570 \\
\end{array}
\]

- Product values $v_j$ and weights $w_j$:

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

Model:

Maximize total value:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to shelf capacity constraints:
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in \{1,2,\ldots,10\}
\]

Integer and nonnegativity constraints:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{1,2,\ldots,10\},\ j \in \{1,2,\ldots,20\}
\]

All coefficients and identifiers are as retrieved above.