##### Decision Variables

Let $x_{ij} \in \{0,1\}$ be 1 if manager $i$ is assigned to project $j$, 0 otherwise.

##### Parameters

Let $M = \{$Manager 1, Manager 2, Manager 3, Manager 4, Manager 5, Manager 6, Manager 7$\}$ (set of managers)

Let $P = \{$Project 1, Project 2, Project 3, Project 4, Project 5, Project 6, Project 7$\}$ (set of projects)

Let $c_{ij}$ be the cost for manager $i$ to complete project $j$, given by the following matrix:

\[
\begin{array}{c|ccccccc}
 & \text{Project 1} & \text{Project 2} & \text{Project 3} & \text{Project 4} & \text{Project 5} & \text{Project 6} & \text{Project 7} \\
\hline
\text{Manager 1} & 2972 & 2727 & 2795 & 2922 & 1302 & 2489 & 1533 \\
\text{Manager 2} & 1094 & 2158 & 2990 & 1844 & 2887 & 2021 & 2288 \\
\text{Manager 3} & 2133 & 1675 & 2422 & 2639 & 1033 & 2261 & 1695 \\
\text{Manager 4} & 1951 & 2309 & 2070 & 2802 & 2328 & 1313 & 2434 \\
\text{Manager 5} & 1269 & 2153 & 1296 & 2685 & 2627 & 1610 & 1641 \\
\text{Manager 6} & 1220 & 1192 & 2907 & 2622 & 2595 & 1261 & 2384 \\
\text{Manager 7} & 1286 & 1659 & 1179 & 1348 & 1420 & 2862 & 1959 \\
\end{array}
\]

##### Objective Function

\[
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]

##### Constraints

1. Each manager is assigned to exactly one project:
   \[
   \sum_{j \in P} x_{ij} = 1, \quad \forall i \in M
   \]

2. Each project is assigned to exactly one manager:
   \[
   \sum_{i \in M} x_{ij} = 1, \quad \forall j \in P
   \]

3. Binary assignment variables:
   \[
   x_{ij} \in \{0,1\}, \quad \forall i \in M,\, j \in P
   \]

##### Retrieved Information

Managers: Manager 1, Manager 2, Manager 3, Manager 4, Manager 5, Manager 6, Manager 7

Projects: Project 1, Project 2, Project 3, Project 4, Project 5, Project 6, Project 7

Cost matrix $c_{ij}$ as shown above.