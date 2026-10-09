##### Sets and Parameters

Let $S = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves, and $P = \{1,2,\ldots,20\}$ be the set of products.

Shelf capacities:
\[
\begin{align*}
C_1 &= 750 \\
C_2 &= 820 \\
C_3 &= 570 \\
C_4 &= 800 \\
C_5 &= 550 \\
C_6 &= 900 \\
C_7 &= 650 \\
C_8 &= 800 \\
C_9 &= 850 \\
C_{10} &= 900 \\
\end{align*}
\]

Product values and weights:
\[
\begin{array}{cccc}
\text{Product} & v_j & w_j \\
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

##### Decision Variables

$x_{ij} \geq 0$: Number of units of product $j \in P$ placed on shelf $i \in S$ (continuous or integer, as appropriate).

##### Objective Function

\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

1. **Shelf capacity constraints** (for each shelf $i \in S$):
   \[
   \sum_{j \in P} w_j x_{ij} \leq C_i
   \]
2. **Nonnegativity**:
   \[
   x_{ij} \geq 0 \qquad \forall i \in S,\, j \in P
   \]

##### Explicit Data

- $S = \{1,2,3,4,5,6,7,8,9,10\}$
- $P = \{1,2,\ldots,20\}$
- Shelf capacities:
  - $C_1 = 750$, $C_2 = 820$, $C_3 = 570$, $C_4 = 800$, $C_5 = 550$, $C_6 = 900$, $C_7 = 650$, $C_8 = 800$, $C_9 = 850$, $C_{10} = 900$
- Product values and weights:
  - $v = [55, 75, 65, 60, 80, 90, 40, 100, 55, 75, 110, 50, 60, 120, 70, 110, 50, 60, 120, 100]$
  - $w = [10, 20, 5, 15, 25, 35, 45, 55, 65, 20, 18, 28, 8, 28, 25, 40, 55, 70, 85, 100]$

##### Complete Mathematical Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i \qquad \forall i=1,\ldots,10 \\
& x_{ij} \geq 0 \qquad \forall i=1,\ldots,10;\; j=1,\ldots,20
\end{align*}
\]

Where all parameters are as listed above.