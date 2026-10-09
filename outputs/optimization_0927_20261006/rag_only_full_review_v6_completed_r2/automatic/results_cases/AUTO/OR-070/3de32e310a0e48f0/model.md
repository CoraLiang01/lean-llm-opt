Let:
- M = {1, 2, 3, 4, 5, 6, 7} be the set of managers, corresponding to Manager 1 through Manager 7.
- P = {1, 2, 3, 4, 5, 6, 7} be the set of projects, corresponding to Project 1 through Project 7.

Define binary variables:
- \( x_{ij} = \begin{cases} 1 & \text{if manager } i \text{ is assigned to project } j \\ 0 & \text{otherwise} \end{cases} \)
  for all \( i \in M, j \in P \).

Let the cost matrix \( C = [c_{ij}] \) be:

\[
C = \begin{bmatrix}
2972 & 2727 & 2795 & 2922 & 1302 & 2489 & 1533 \\
1094 & 2158 & 2990 & 1844 & 2887 & 2021 & 2288 \\
2133 & 1675 & 2422 & 2639 & 1033 & 2261 & 1695 \\
1951 & 2309 & 2070 & 2802 & 2328 & 1313 & 2434 \\
1269 & 2153 & 1296 & 2685 & 2627 & 1610 & 1641 \\
1220 & 1192 & 2907 & 2622 & 2595 & 1261 & 2384 \\
1286 & 1659 & 1179 & 1348 & 1420 & 2862 & 1959 \\
\end{bmatrix}
\]

where \( c_{ij} \) is the cost for manager \( i \) to complete project \( j \).

The mathematical model is:

\[
\begin{align*}
\text{Minimize} \quad & \sum_{i=1}^7 \sum_{j=1}^7 c_{ij} x_{ij} \\
\text{subject to} \quad
& \sum_{i=1}^7 x_{ij} = 1 \quad \forall j \in P \quad \text{(each project assigned to exactly one manager)} \\
& \sum_{j=1}^7 x_{ij} \leq 1 \quad \forall i \in M \quad \text{(each manager assigned to at most one project)} \\
& x_{ij} \in \{0,1\} \quad \forall i \in M, j \in P
\end{align*}
\]

Where:
- \( x_{ij} \) are binary assignment variables.
- \( c_{ij} \) are the costs as given in the matrix above.

All assignments are eligible; there are no additional qualification or availability constraints in the data or query. The model enforces that each project is assigned to exactly one manager, and each manager is responsible for at most one project, as required. The objective is to minimize the total cost of assignments.