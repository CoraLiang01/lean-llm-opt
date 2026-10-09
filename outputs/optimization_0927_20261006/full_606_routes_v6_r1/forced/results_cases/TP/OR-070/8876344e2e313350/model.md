##### Sets

Let $M = \{\text{Manager 1}, \text{Manager 2}, \text{Manager 3}, \text{Manager 4}, \text{Manager 5}, \text{Manager 6}, \text{Manager 7}\}$ be the set of managers.

Let $P = \{\text{Project 1}, \text{Project 2}, \text{Project 3}, \text{Project 4}, \text{Project 5}, \text{Project 6}, \text{Project 7}\}$ be the set of projects.

##### Parameters

Let $c_{mp}$ be the cost for manager $m$ to complete project $p$, as given below:

\[
\begin{array}{l|ccccccc}
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

##### Decision Variables

For each $m \in M$, $p \in P$:

\[
x_{mp} =
\begin{cases}
1 & \text{if manager } m \text{ is assigned to project } p \\
0 & \text{otherwise}
\end{cases}
\]

##### Objective Function

\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} x_{mp}
\]

##### Constraints

1. Each project is assigned to exactly one manager:
   \[
   \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
   \]

2. Each manager is assigned to exactly one project:
   \[
   \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
   \]

3. Binary assignment variables:
   \[
   x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
   \]