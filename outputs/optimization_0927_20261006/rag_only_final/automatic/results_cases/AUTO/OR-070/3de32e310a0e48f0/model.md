Let:
- M = {1, 2, 3, 4, 5, 6, 7} be the set of managers (Manager 1 to Manager 7)
- P = {1, 2, 3, 4, 5, 6, 7} be the set of projects (Project 1 to Project 7)
- c_{ij} be the cost for manager i to complete project j, as given in the matrix below
- x_{ij} ∈ {0,1} is a binary variable: x_{ij} = 1 if manager i is assigned to project j, 0 otherwise

Cost matrix C = [c_{ij}] (rows: managers, columns: projects):

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

Mathematical Model:

Decision variables:
\[
x_{ij} = \begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
\]
for all \( i \in M, j \in P \).

Objective:
\[
\min \sum_{i=1}^{7} \sum_{j=1}^{7} c_{ij} x_{ij}
\]

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j=1}^{7} x_{ij} = 1 \quad \forall i \in M
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i=1}^{7} x_{ij} = 1 \quad \forall j \in P
\]

3. Binary variables:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M, j \in P
\]

Where:
- Managers: Manager 1, Manager 2, Manager 3, Manager 4, Manager 5, Manager 6, Manager 7
- Projects: Project 1, Project 2, Project 3, Project 4, Project 5, Project 6, Project 7
- Cost matrix C as specified above.

This is a standard assignment problem (linear sum assignment) formulated as an integer program.