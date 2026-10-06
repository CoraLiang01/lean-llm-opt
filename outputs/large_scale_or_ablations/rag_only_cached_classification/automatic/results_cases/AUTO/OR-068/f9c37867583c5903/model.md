Let:
- M = {MA, MB, MC, MD, ME, MF} be the set of managers (indexed by i)
- P = {P1, P2, P3, P4, P5, P6} be the set of projects (indexed by j)
- cij = cost of assigning manager i to project j, as given in the matrix below

Cost matrix C = [cij] (rows: managers, columns: projects):

|      |   P1  |   P2  |   P3  |   P4  |   P5  |   P6  |
|------|-------|-------|-------|-------|-------|-------|
| MA   | 2216  | 1911  | 1661  | 2122  | 1442  | 1442  |
| MB   | 1100  | 1271  | 2764  | 2557  | 1036  | 1036  |
| MC   | 2827  | 2784  | 2206  | 2216  | 2677  | 2677  |
| MD   | 2627  | 1273  | 2610  | 1957  | 1594  | 1594  |
| ME   | 3359  | 1003  | 2554  | 1706  | 2065  | 2065  |
| MF   | 1579  | 2289  | 2368  | 1922  | 2740  | 2740  |

Decision variables:
Let x_{ij} = 
    1 if manager i is assigned to project j,
    0 otherwise.

Mathematical Model:

Minimize total assignment cost:
\[
\text{Minimize} \quad Z = \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]
where c_{ij} is as given in the matrix above.

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j \in P} x_{ij} = 1 \quad \forall i \in M
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in P
\]

3. Binary assignment variables:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M, j \in P
\]

Where:
- M = {MA, MB, MC, MD, ME, MF}
- P = {P1, P2, P3, P4, P5, P6}
- cij as given in the cost matrix above.

This is the required mathematical model for the minimum-cost one-to-one assignment of managers to projects using the provided cost data.