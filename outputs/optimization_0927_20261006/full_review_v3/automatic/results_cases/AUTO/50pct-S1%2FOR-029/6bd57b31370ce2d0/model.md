Let $x_{ij}$ be the number of units of product $j$ placed on display (shelf) $i$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$ for all $i, j$.

Let $S$ be the set of shelves (displays), indexed by ShelfID:
$$
S = \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10\}
$$

Let $P$ be the set of products, indexed in the order given in products.csv:
\[
\begin{array}{ll}
1: & \text{Smartphone} \\
2: & \text{Laptop} \\
3: & \text{Headphones} \\
4: & \text{Camera} \\
5: & \text{Smartwatch} \\
6: & \text{Tablet} \\
7: & \text{Bluetooth Speaker} \\
8: & \text{Keyboard} \\
9: & \text{Mouse} \\
10: & \text{Monitor} \\
11: & \text{Printer} \\
12: & \text{External Hard Drive} \\
13: & \text{Router} \\
14: & \text{Power Bank} \\
15: & \text{Memory Card} \\
16: & \text{USB Flash Drive} \\
17: & \text{Smart Home Hub} \\
18: & \text{Gaming Console} \\
19: & \text{Fitness Tracker} \\
20: & \text{E-Reader} \\
\end{array}
\]

Let $v_j$ be the value of product $j$ and $w_j$ be the weight of product $j$:

\[
\begin{array}{lll}
j & v_j & w_j \\
1 & 200 & 1 \\
2 & 1500 & 5 \\
3 & 100 & 0.5 \\
4 & 800 & 2 \\
5 & 250 & 0.3 \\
6 & 600 & 1.5 \\
7 & 150 & 1 \\
8 & 80 & 0.8 \\
9 & 50 & 0.2 \\
10 & 300 & 3 \\
11 & 400 & 4 \\
12 & 120 & 0.5 \\
13 & 60 & 0.3 \\
14 & 40 & 0.4 \\
15 & 30 & 0.05 \\
16 & 25 & 0.02 \\
17 & 100 & 0.6 \\
18 & 500 & 4 \\
19 & 90 & 0.2 \\
20 & 180 & 0.5 \\
\end{array}
\]

Let $C_i$ be the capacity of shelf $i$:

\[
\begin{array}{ll}
i & C_i \\
1 & 5 \\
2 & 7 \\
3 & 6 \\
4 & 8 \\
5 & 5.5 \\
6 & 9 \\
7 & 6.5 \\
8 & 7.5 \\
9 & 8.2 \\
10 & 5.7 \\
\end{array}
\]

The mathematical model is:

\[
\textbf{Objective:} \quad \max \sum_{i \in S} \sum_{j=1}^{20} v_j x_{ij}
\]

\[
\textbf{Subject to:}
\]

\[
\sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i \in S
\]

\[
\sum_{i \in S} x_{i1} \geq 5
\]

\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S, \; j = 1, \ldots, 20
\]

Where:
- $x_{ij}$: number of units of product $j$ placed on shelf $i$
- $v_j$: value of product $j$ (see table above)
- $w_j$: weight of product $j$ (see table above)
- $C_i$: capacity of shelf $i$ (see table above)
- $S$: set of shelves $\{1,2,3,4,5,6,7,8,9,10\}$

All coefficients and identifiers are as retrieved and in original order.