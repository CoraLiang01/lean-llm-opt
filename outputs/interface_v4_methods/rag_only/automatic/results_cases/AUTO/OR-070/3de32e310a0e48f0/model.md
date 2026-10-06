Let there be 7 managers (indexed i = 1,...,7) and 7 projects (indexed j = 1,...,7). Let c_{ij} denote the cost for Manager i to complete Project j, as given in the matrix below:

Cost matrix C = [c_{ij}]:

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

Let x_{ij} be a binary variable:
- x_{ij} = 1 if Manager i is assigned to Project j
- x_{ij} = 0 otherwise

Mathematical Model:

Minimize total cost:
\[
\min \sum_{i=1}^{7} \sum_{j=1}^{7} c_{ij} x_{ij}
\]

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j=1}^{7} x_{ij} = 1 \quad \forall i = 1,...,7
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i=1}^{7} x_{ij} = 1 \quad \forall j = 1,...,7
\]

3. Binary assignment variables:
\[
x_{ij} \in \{0,1\} \quad \forall i = 1,...,7; \; j = 1,...,7
\]

Where:
- Managers: [Manager 1, Manager 2, Manager 3, Manager 4, Manager 5, Manager 6, Manager 7]
- Projects: [Project 1, Project 2, Project 3, Project 4, Project 5, Project 6, Project 7]
- Cost matrix C as given above.

This model ensures each manager is assigned to exactly one project, each project is assigned to exactly one manager, and the total cost is minimized according to the provided cost data.