Let:
- M = {MA, MB, MC, MD, ME, MF} (set of managers, indexed by i)
- P = {P1, P2, P3, P4, P5, P6} (set of projects, indexed by j)
- C[i,j]: cost of assigning manager i to project j, as given below
- x[i,j]: binary variable, 1 if manager i is assigned to project j, 0 otherwise

Cost matrix C[i,j]:
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
where the rows correspond to [MA, MB, MC, MD, ME, MF] and columns to [P1, P2, P3, P4, P5, P6].

Eligibility mask E[i,j]:
\[
E[i,j] = 1 \quad \forall i \in M, j \in P
\]
(all pairs are eligible; no exclusions)

Mathematical Model:

Variables:
- \( x[i,j] \in \{0,1\} \) for all \( i \in M, j \in P \)

Objective:
\[
\min \sum_{i \in M} \sum_{j \in P} C[i,j] \cdot x[i,j]
\]

Subject to:
1. Each manager is assigned to exactly one project:
   \[
   \sum_{j \in P} x[i,j] = 1 \quad \forall i \in M
   \]
2. Each project is assigned to exactly one manager:
   \[
   \sum_{i \in M} x[i,j] = 1 \quad \forall j \in P
   \]
3. Eligibility mask (all pairs eligible, so no exclusions):
   \[
   x[i,j] \in \{0,1\} \quad \forall i \in M, j \in P
   \]

All parameters:
- Managers: MA, MB, MC, MD, ME, MF
- Projects: P1, P2, P3, P4, P5, P6
- Cost matrix C[i,j] as above
- Eligibility mask E[i,j] = 1 for all i, j

This is the complete mathematical model for the minimum-cost one-to-one assignment of managers to projects, as specified by the data and the query.