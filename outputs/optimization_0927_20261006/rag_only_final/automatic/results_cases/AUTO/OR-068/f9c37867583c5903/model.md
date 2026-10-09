Let M = {MA, MB, MC, MD, ME, MF} be the set of managers, and P = {P1, P2, P3, P4, P5, P6} be the set of projects.

Let c_{ij} denote the cost of assigning manager i to project j, as given in the following cost matrix:

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

where the rows correspond to managers [MA, MB, MC, MD, ME, MF] and the columns correspond to projects [P1, P2, P3, P4, P5, P6].

Define binary decision variables:
\[
x_{ij} = \begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
\]
for all \( i \in M \), \( j \in P \).

The mathematical model is:

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
- \( c_{ij} \) is given by the matrix above.
- \( x_{ij} \) are binary variables as defined.

This model finds the minimum-cost one-to-one assignment of managers to projects, using the exact costs from the provided CSV data.