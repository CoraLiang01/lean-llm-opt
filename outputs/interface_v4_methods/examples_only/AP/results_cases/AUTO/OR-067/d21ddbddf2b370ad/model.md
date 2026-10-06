Let:
- M = {MA, MB, MC} be the set of managers
- P = {P1, P2, P3} be the set of projects

Let cij denote the cost for manager i to complete project j, given by the following cost matrix C:

C = 
|      | P1   | P2   | P3   |
|------|------|------|------|
| MA   | 3000 | 3200 | 3100 |
| MB   | 2800 | 3300 | 2900 |
| MC   | 2900 | 3100 | 3000 |

Let x_{ij} be a binary decision variable:
- x_{ij} = 1 if manager i is assigned to project j
- x_{ij} = 0 otherwise

Mathematical Model:

Minimize total cost:
\[
\text{Minimize} \quad Z = 3000x_{MA,P1} + 3200x_{MA,P2} + 3100x_{MA,P3} + 2800x_{MB,P1} + 3300x_{MB,P2} + 2900x_{MB,P3} + 2900x_{MC,P1} + 3100x_{MC,P2} + 3000x_{MC,P3}
\]

Subject to:
1. Each manager is assigned to exactly one project:
\[
x_{MA,P1} + x_{MA,P2} + x_{MA,P3} = 1 \\
x_{MB,P1} + x_{MB,P2} + x_{MB,P3} = 1 \\
x_{MC,P1} + x_{MC,P2} + x_{MC,P3} = 1
\]

2. Each project is assigned to exactly one manager:
\[
x_{MA,P1} + x_{MB,P1} + x_{MC,P1} = 1 \\
x_{MA,P2} + x_{MB,P2} + x_{MC,P2} = 1 \\
x_{MA,P3} + x_{MB,P3} + x_{MC,P3} = 1
\]

3. Binary constraints:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M, j \in P
\]

Where:
- M = {MA, MB, MC}
- P = {P1, P2, P3}
- C = \begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 \\
\end{bmatrix}

This model assigns each manager to exactly one project and each project to exactly one manager, minimizing the total assignment cost as specified in the CSV data.