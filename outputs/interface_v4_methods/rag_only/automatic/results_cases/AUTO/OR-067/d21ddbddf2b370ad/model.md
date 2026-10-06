Let:
- M = {MA, MB, MC} be the set of managers.
- P = {P1, P2, P3} be the set of projects.
- cij be the cost for manager i to complete project j, as given in the cost matrix below.
- xij be a binary decision variable, where xij = 1 if manager i is assigned to project j, and 0 otherwise.

Cost matrix C (with cij values):

|        | P1   | P2   | P3   |
|--------|------|------|------|
| MA     | 3000 | 3200 | 3100 |
| MB     | 2800 | 3300 | 2900 |
| MC     | 2900 | 3100 | 3000 |

Mathematical Model:

Variables:
- For each manager i ∈ M and project j ∈ P, define xij ∈ {0, 1}

Objective:
Minimize the total cost:
\[
\text{Minimize} \quad Z = 3000x_{MA,P1} + 3200x_{MA,P2} + 3100x_{MA,P3} + 2800x_{MB,P1} + 3300x_{MB,P2} + 2900x_{MB,P3} + 2900x_{MC,P1} + 3100x_{MC,P2} + 3000x_{MC,P3}
\]

Subject to:

1. Each manager is assigned to exactly one project:
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
x_{MA,P1} + x_{MB,P1} + x_{MC,P1} = 1
\]
\[
x_{MA,P2} + x_{MB,P2} + x_{MC,P2} = 1
\]
\[
x_{MA,P3} + x_{MB,P3} + x_{MC,P3} = 1
\]

3. Binary assignment variables:
\[
x_{ij} \in \{0, 1\} \quad \forall i \in M, j \in P
\]

Summary:
- Decision variables: x_{ij} for all i ∈ {MA, MB, MC}, j ∈ {P1, P2, P3}
- Cost matrix C as above
- Objective: Minimize total assignment cost
- Constraints: Each manager and each project is assigned exactly once; variables are binary

This is a standard assignment problem (linear sum assignment), fully specified with the provided data.