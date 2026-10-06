Let $N = \{\text{Depot}, A, B, C\}$ be the set of nodes (Depot and three customers).

Let $d_{ij}$ be the distance from node $i$ to node $j$, for $i, j \in N$, as given below:

\[
\begin{array}{c|cccc}
 & \text{Depot} & A & B & C \\
\hline
\text{Depot} & 0 & 28 & 41 & 63 \\
A & 28 & 0 & 27 & 87 \\
B & 41 & 27 & 0 & 81 \\
C & 63 & 87 & 81 & 0 \\
\end{array}
\]

Define binary decision variables:
\[
x_{ij} = \begin{cases}
1 & \text{if the van travels directly from node } i \text{ to node } j \\
0 & \text{otherwise}
\end{cases}
\qquad \forall i, j \in N,\, i \neq j
\]

Auxiliary variables for subtour elimination (Miller-Tucker-Zemlin formulation):
\[
u_i \in \mathbb{Z},\quad 1 \leq u_i \leq 3,\quad \forall i \in \{A, B, C\}
\]

Objective:
\[
\min \sum_{i \in N} \sum_{\substack{j \in N \\ j \neq i}} d_{ij} x_{ij}
\]

Subject to:

1. Each node is departed exactly once:
\[
\sum_{\substack{j \in N \\ j \neq i}} x_{ij} = 1 \qquad \forall i \in N
\]

2. Each node is arrived at exactly once:
\[
\sum_{\substack{i \in N \\ i \neq j}} x_{ij} = 1 \qquad \forall j \in N
\]

3. Subtour elimination (for all $i, j \in \{A, B, C\}$, $i \neq j$):
\[
u_i - u_j + 3 x_{ij} \leq 2
\]

4. Binary and integer restrictions:
\[
x_{ij} \in \{0,1\} \qquad \forall i, j \in N,\, i \neq j
\]
\[
u_i \in \mathbb{Z},\ 1 \leq u_i \leq 3 \qquad \forall i \in \{A, B, C\}
\]

Where the distance coefficients $d_{ij}$ are:

\[
\begin{array}{c|cccc}
 & \text{Depot} & A & B & C \\
\hline
\text{Depot} & 0 & 28 & 41 & 63 \\
A & 28 & 0 & 27 & 87 \\
B & 41 & 27 & 0 & 81 \\
C & 63 & 87 & 81 & 0 \\
\end{array}
\]

The optimal sequence of visits is the order of nodes corresponding to the tour with minimum total distance, as determined by solving the above model.