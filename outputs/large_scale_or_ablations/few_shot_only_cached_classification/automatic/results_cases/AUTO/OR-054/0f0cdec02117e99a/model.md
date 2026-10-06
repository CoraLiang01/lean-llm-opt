Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

Objective:
\[
\max \sum_{i=1}^{10} \sum_{j=1}^{20} v_j \cdot x_{ij}
\]
where $v_j$ is the Value of product $j$ from products.csv.

Subject to:

For each shelf $i$ (ShelfID from 1 to 10):
\[
\sum_{j=1}^{20} w_j \cdot x_{ij} \leq C_i
\]
where $w_j$ is the Weight of product $j$ from products.csv, and $C_i$ is the Capacity of shelf $i$ from capacity.csv.

Variable domains:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in \{1,\ldots,10\},\ j \in \{1,\ldots,20\}
\]

Numerical Data:

Shelf Capacities (from capacity.csv, in source order):
\[
\begin{array}{ll}
C_1 = 750 & C_6 = 900 \\
C_2 = 820 & C_7 = 650 \\
C_3 = 570 & C_8 = 800 \\
C_4 = 800 & C_9 = 850 \\
C_5 = 550 & C_{10} = 900 \\
\end{array}
\]

Product Values and Weights (from products.csv, in source order):

\[
\begin{array}{cccc}
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

Complete Model:

\[
\begin{align*}
\max\ & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad
& \sum_{j=1}^{20} w_j x_{ij} \leq C_i \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i=1,\ldots,10,\ j=1,\ldots,20
\end{align*}
\]

Where all $v_j$, $w_j$, and $C_i$ are as listed above, and all indices and identifiers are preserved in source order.