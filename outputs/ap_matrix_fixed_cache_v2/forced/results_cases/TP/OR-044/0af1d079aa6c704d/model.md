##### Sets

Let $S = \{1,2,3,4,5,6,7,8\}$ be the set of sections (SectionID from capacity.csv).

Let $P = \{1,2,3,4,5,6,7,8,9,10\}$ be the set of products (ProductName from products.csv).

##### Parameters

For each section $i \in S$:

- $C_i$ = capacity of section $i$ (from capacity.csv):

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

For each product $j \in P$:

- $v_j$ = value (price) of product $j$ (from products.csv)
- $w_j$ = weight (shelf space requirement) of product $j$ (from products.csv)

\[
\begin{align*}
v_1 &= 10, & w_1 &= 2 \\
v_2 &= 15, & w_2 &= 3 \\
v_3 &= 8,  & w_3 &= 1 \\
v_4 &= 12, & w_4 &= 2 \\
v_5 &= 20, & w_5 &= 4 \\
v_6 &= 25, & w_6 &= 5 \\
v_7 &= 5,  & w_7 &= 1 \\
v_8 &= 30, & w_8 &= 6 \\
v_9 &= 18, & w_9 &= 3 \\
v_{10} &= 22, & w_{10} &= 4 \\
\end{align*}
\]

##### Decision Variables

For each section $i \in S$ and product $j \in P$:

- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ to be placed in section $i$ (integer, $\geq 0$).

##### Objective

\[
\max \sum_{i \in S} \sum_{j \in P} v_j x_{ij}
\]

##### Constraints

For each section $i \in S$:

\[
\sum_{j \in P} w_j x_{ij} \leq C_i
\]

For all $i \in S$, $j \in P$:

\[
x_{ij} \in \mathbb{Z}_{\geq 0}
\]

##### Complete Model

\[
\begin{align*}
\max\quad & \sum_{i=1}^8 \sum_{j=1}^{10} v_j x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{10} w_j x_{ij} \leq C_i, \quad \forall i=1,\ldots,8 \\
& x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i=1,\ldots,8,\ j=1,\ldots,10 \\
\end{align*}
\]

Where:

- $C_1 = 100$, $C_2 = 150$, $C_3 = 120$, $C_4 = 130$, $C_5 = 90$, $C_6 = 110$, $C_7 = 160$, $C_8 = 140$
- $v_1 = 10$, $w_1 = 2$; $v_2 = 15$, $w_2 = 3$; $v_3 = 8$, $w_3 = 1$; $v_4 = 12$, $w_4 = 2$; $v_5 = 20$, $w_5 = 4$; $v_6 = 25$, $w_6 = 5$; $v_7 = 5$, $w_7 = 1$; $v_8 = 30$, $w_8 = 6$; $v_9 = 18$, $w_9 = 3$; $v_{10} = 22$, $w_{10} = 4$