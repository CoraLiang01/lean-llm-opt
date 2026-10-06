Let:
- There are 7 managers (indexed by i = 1,...,7) and 7 projects (indexed by j = 1,...,7).
- Let x_{ij} be a binary decision variable, where x_{ij} = 1 if manager i is assigned to project j, and 0 otherwise.
- Let C_{ij} be the cost for manager i to complete project j, as given in the cost matrix below.

Cost matrix C (rows: managers, columns: projects):

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

The mathematical model is:

Decision variables:
\[
x_{ij} = 
\begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
\]

Objective:
\[
\text{Minimize} \quad Z = \sum_{i=1}^{7} \sum_{j=1}^{7} C_{ij} x_{ij}
\]
where C_{ij} is as given in the matrix above.

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j=1}^{7} x_{ij} = 1 \quad \forall i = 1,...,7
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i=1}^{7} x_{ij} = 1 \quad \forall j = 1,...,7
\]

3. Binary constraints:
\[
x_{ij} \in \{0,1\} \quad \forall i, j
\]

Where:
- Managers: Manager 1, Manager 2, ..., Manager 7
- Projects: Project 1, Project 2, ..., Project 7

This is a standard assignment problem (linear integer programming) with the objective to minimize the total cost of assigning managers to projects, using the provided cost matrix.