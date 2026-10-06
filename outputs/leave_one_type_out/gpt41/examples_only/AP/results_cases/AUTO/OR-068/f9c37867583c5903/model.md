Let:
- M = {MA, MB, MC, MD, ME, MF} be the set of managers (indexed by i = 1,...,6)
- P = {P1, P2, P3, P4, P5, P6} be the set of projects (indexed by j = 1,...,6)
- c_{ij} be the cost of assigning manager i to project j, as given in the cost matrix below.
- x_{ij} ∈ {0,1} be a binary decision variable, where x_{ij} = 1 if manager i is assigned to project j, and 0 otherwise.

The cost matrix C = [c_{ij}] is:

\[
C = \begin{bmatrix}
2216 & 1911 & 1661 & 2122 & 1442 & 1442 \\
1100 & 1271 & 2764 & 2557 & 1036 & 1036 \\
2827 & 2784 & 2206 & 2216 & 2677 & 2677 \\
2627 & 1273 & 2610 & 1957 & 1594 & 1594 \\
3359 & 1003 & 2554 & 1706 & 2065 & 2065 \\
1579 & 2289 & 2368 & 1922 & 2740 & 2740 \\
\end{bmatrix}
\]

where the rows correspond to managers (MA, MB, MC, MD, ME, MF) and the columns correspond to projects (P1, P2, P3, P4, P5, P6).

Mathematical Model:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^{6} \sum_{j=1}^{6} c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{j=1}^{6} x_{ij} = 1 \quad \forall i \in \{1,2,3,4,5,6\} \quad \text{(each manager assigned to one project)} \\
& \sum_{i=1}^{6} x_{ij} = 1 \quad \forall j \in \{1,2,3,4,5,6\} \quad \text{(each project assigned to one manager)} \\
& x_{ij} \in \{0,1\} \quad \forall i, j
\end{align*}
\]

Where:
- \( c_{ij} \) is as given in the matrix above.
- \( x_{ij} \) are binary variables indicating assignments.

This is a classical assignment problem (minimum-cost bipartite matching) with the objective to minimize the total assignment cost, ensuring a one-to-one assignment between managers and projects.