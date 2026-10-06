Let:
- M = {MA, MB, MC} be the set of managers (indexed by i)
- P = {P1, P2, P3} be the set of projects (indexed by j)
- c_{ij} be the cost for manager i to complete project j, given by the following cost matrix:

\[
C = \begin{bmatrix}
c_{MA,P1} & c_{MA,P2} & c_{MA,P3} \\
c_{MB,P1} & c_{MB,P2} & c_{MB,P3} \\
c_{MC,P1} & c_{MC,P2} & c_{MC,P3} \\
\end{bmatrix}
=
\begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 \\
\end{bmatrix}
\]

Define binary decision variables:
\[
x_{ij} = 
\begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
\]

The mathematical model is:

\[
\textbf{Minimize:} \quad Z = \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M \quad \text{(each manager assigned to exactly one project)}
\]
\[
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P \quad \text{(each project assigned to exactly one manager)}
\]
\[
x_{ij} \in \{0,1\} \quad \forall i \in M, \forall j \in P
\]

Where:
- M = {MA, MB, MC}
- P = {P1, P2, P3}
- Cost matrix C as above.

This is the mathematical model for the optimal assignment of managers to projects, minimizing total cost, based on the provided cost matrix.