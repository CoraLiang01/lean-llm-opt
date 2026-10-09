Let:
- M = {MA, MB, MC} be the set of managers
- P = {P1, P2, P3} be the set of projects

Parameters:
- c_{ij}: cost for manager i to complete project j, given by the following matrix:

\[
C = \begin{bmatrix}
c_{MA,P1} & c_{MA,P2} & c_{MA,P3} \\
c_{MB,P1} & c_{MB,P2} & c_{MB,P3} \\
c_{MC,P1} & c_{MC,P2} & c_{MC,P3}
\end{bmatrix}
=
\begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000
\end{bmatrix}
\]

Decision variables:
- \( x_{ij} \in \{0,1\} \) for all \( i \in M, j \in P \), where \( x_{ij} = 1 \) if manager i is assigned to project j, 0 otherwise.

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{j \in P} x_{ij} = 1 \quad \forall i \in M \quad \text{(each manager assigned to exactly one project)} \\
& \sum_{i \in M} x_{ij} = 1 \quad \forall j \in P \quad \text{(each project assigned to exactly one manager)} \\
& x_{ij} \in \{0,1\} \quad \forall i \in M, j \in P
\end{align*}
\]

Where:
- \( c_{MA,P1} = 3000, c_{MA,P2} = 3200, c_{MA,P3} = 3100 \)
- \( c_{MB,P1} = 2800, c_{MB,P2} = 3300, c_{MB,P3} = 2900 \)
- \( c_{MC,P1} = 2900, c_{MC,P2} = 3100, c_{MC,P3} = 3000 \)

This model finds the optimal one-to-one assignment of managers to projects that minimizes the total cost, using the cost matrix provided in the CSV file.