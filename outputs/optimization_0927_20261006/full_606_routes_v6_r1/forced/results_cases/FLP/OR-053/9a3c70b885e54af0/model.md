##### Sets and Indices

- Let $S = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of shelves, indexed by $i$.
- Let $P = \{1,2,3,\ldots,20\}$ be the set of products, indexed by $j$.

##### Parameters

- Shelf capacities (from capacity.csv):

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

- Product values and weights (from products.csv):

\[
\begin{array}{cccc}
\text{Product} & v_j & w_j \\
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

##### Decision Variables

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on shelf $i$.

##### Objective Function

\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

1. **Shelf capacity constraints:** For each shelf $i \in S$,
   \[
   \sum_{j \in P} w_j x_{ij} \leq C_i
   \]

2. **Nonnegativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in S,\, j \in P
   \]

##### Parameters (explicit listing)

- $S = \{1,2,3,4,5,6,7,8,9,10\}$
- $P = \{1,2,3,\ldots,20\}$
- $C = [500, 700, 600, 800, 550, 900, 650, 750, 820, 570]$
- $v = [50, 70, 30, 60, 80, 90, 40, 100, 55, 75, 65, 95, 45, 85, 70, 110, 50, 60, 120, 100]$
- $w = [10, 20, 5, 15, 25, 30, 12, 35, 10, 20, 18, 28, 8, 22, 25, 40, 14, 16, 50, 30]$

##### Complete Mathematical Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^{10} \sum_{j=1}^{20} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{20} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,10 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,10,\; j=1,\ldots,20
\end{align*}
\]

where $v_j$ and $w_j$ are as listed above, and $C_i$ are the shelf capacities as listed.