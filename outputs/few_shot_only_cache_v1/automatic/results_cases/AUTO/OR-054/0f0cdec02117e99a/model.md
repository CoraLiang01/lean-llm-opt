Let $x_{ij}$ be the number of units of product $j$ placed on shelf $i$, where $i \in \{1,2,\ldots,10\}$ (ShelfID from capacity.csv) and $j \in \{1,2,\ldots,20\}$ (ProductName from products.csv).

Parameters:
- $v_j$: Value of product $j$ (from products.csv)
- $w_j$: Weight of product $j$ (from products.csv)
- $C_i$: Capacity of shelf $i$ (from capacity.csv)

Data:

Shelf capacities (from capacity.csv):

\[
\begin{array}{ll}
\text{ShelfID} & \text{Capacity} \\
1 & 750 \\
2 & 820 \\
3 & 570 \\
4 & 800 \\
5 & 550 \\
6 & 900 \\
7 & 650 \\
8 & 800 \\
9 & 850 \\
10 & 900 \\
\end{array}
\]

Product values and weights (from products.csv):

\[
\begin{array}{lll}
\text{ProductName} & v_j & w_j \\
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
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i = 1,\ldots,10;\; j = 1,\ldots,20
\]

Where:
- $v_j$ and $w_j$ are as given above for each product $j$.
- $C_i$ is as given above for each shelf $i$.

All identifiers and coefficients are preserved from the supplied data.