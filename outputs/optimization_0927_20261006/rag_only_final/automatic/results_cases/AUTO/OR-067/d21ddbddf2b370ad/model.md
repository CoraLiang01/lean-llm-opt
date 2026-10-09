Let:
- M = {MA, MB, MC} be the set of managers.
- P = {P1, P2, P3} be the set of projects.
- c_{ij} be the cost for manager i to complete project j, as given in the table below.
- x_{ij} = 1 if manager i is assigned to project j, 0 otherwise.

Cost matrix (c_{ij}):
\[
\begin{array}{c|ccc}
      & P1   & P2   & P3   \\
\hline
MA    & 3000 & 3200 & 3100 \\
MB    & 2800 & 3300 & 2900 \\
MC    & 2900 & 3100 & 3000 \\
\end{array}
\]

Mathematical Model:

Decision Variables:
\[
x_{ij} = 
\begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
\]
for all \( i \in M, j \in P \).

Objective:
\[
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]
That is,
\[
\min \Big( 3000x_{MA,P1} + 3200x_{MA,P2} + 3100x_{MA,P3} + 2800x_{MB,P1} + 3300x_{MB,P2} + 2900x_{MB,P3} + 2900x_{MC,P1} + 3100x_{MC,P2} + 3000x_{MC,P3} \Big)
\]

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
\]
That is,
\[
x_{MA,P1} + x_{MA,P2} + x_{MA,P3} = 1
\]
\[
x_{MB,P1} + x_{MB,P2} + x_{MB,P3} = 1
\]
\[
x_{MC,P1} + x_{MC,P2} + x_{MC,P3} = 1
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
\]
That is,
\[
x_{MA,P1} + x_{MB,P1} + x_{MC,P1} = 1
\]
\[
x_{MA,P2} + x_{MB,P2} + x_{MC,P2} = 1
\]
\[
x_{MA,P3} + x_{MB,P3} + x_{MC,P3} = 1
\]

3. Binary variables:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M, j \in P
\]

Parameters:
- Managers: M = [MA, MB, MC]
- Projects: P = [P1, P2, P3]
- Cost matrix:
  - c_{MA,P1} = 3000, c_{MA,P2} = 3200, c_{MA,P3} = 3100
  - c_{MB,P1} = 2800, c_{MB,P2} = 3300, c_{MB,P3} = 2900
  - c_{MC,P1} = 2900, c_{MC,P2} = 3100, c_{MC,P3} = 3000

This model assigns each manager to exactly one project and each project to exactly one manager, minimizing the total assignment cost.