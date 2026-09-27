Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

Parameters:

- $v_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight of product $j$ (from products.csv)
- $C_i$: Capacity of shelf $i$ (from capacity.csv)

Data:

Shelf capacities (from capacity.csv):

\[
\begin{array}{ll}
C_1 = 750 & C_6 = 900 \\
C_2 = 820 & C_7 = 650 \\
C_3 = 570 & C_8 = 800 \\
C_4 = 800 & C_9 = 850 \\
C_5 = 550 & C_{10} = 900 \\
\end{array}
\]

Product values and weights (from products.csv):

\[
\begin{array}{lll}
j & v_j & w_j \\
1 & 55 & 10 \\
2 & 75 & 20 \\
3 & 65 & 5 \\
4 & 60 & 15 \\
5 & 80 & 25 \\
6 & 90 & 35 \\
7 & 40 & 45 \\
8 & 100 & 55 \\
9 & 55 & 65 \\
10 & 75 & 20 \\
11 & 110 & 18 \\
12 & 50 & 28 \\
13 & 60 & 8 \\
14 & 120 & 28 \\
15 & 70 & 25 \\
16 & 110 & 40 \\
17 & 50 & 55 \\
18 & 60 & 70 \\
19 & 120 & 85 \\
20 & 100 & 100 \\
\end{array}
\]

Model:

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij}
\]

Subject to, for each shelf $i$:
\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i = 1,\ldots,10
\]

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\ j = 1,\ldots,20
\]

Where all coefficients and identifiers are as listed above.