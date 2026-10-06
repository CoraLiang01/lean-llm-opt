Let us define the following sets and parameters based on the CSV data:

Sets:
- Let M = {MA, MB, MC} be the set of managers.
- Let P = {P1, P2, P3} be the set of projects.

Parameters:
- Let c_{mp} denote the cost for manager m ∈ M to complete project p ∈ P, as given in the cost matrix:

\[
C = \begin{bmatrix}
c_{MA,P1} & c_{MA,P2} & c_{MA,P3} \\
c_{MB,P1} & c_{MB,P2} & c_{MB,P3} \\
c_{MC,P1} & c_{MC,P2} & c_{MC,P3} \\
\end{bmatrix}
=
\begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 \\
\end{bmatrix}
\]

Decision Variables:
- Let x_{mp} = 1 if manager m is assigned to project p, and 0 otherwise, for all m ∈ M, p ∈ P.

Mathematical Model:

Objective:
\[
\text{Minimize} \quad Z = \sum_{m \in M} \sum_{p \in P} c_{mp} x_{mp}
\]
That is,
\[
\text{Minimize} \quad Z = 3000x_{MA,P1} + 3200x_{MA,P2} + 3100x_{MA,P3} + 2800x_{MB,P1} + 3300x_{MB,P2} + 2900x_{MB,P3} + 2900x_{MC,P1} + 3100x_{MC,P2} + 3000x_{MC,P3}
\]

Subject to:

1. Each manager is assigned to exactly one project:
\[
\sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
\]
Explicitly:
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
\sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
\]
Explicitly:
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
x_{mp} \in \{0,1\} \quad \forall m \in M, p \in P
\]

Summary:
- Sets: M = {MA, MB, MC}, P = {P1, P2, P3}
- Cost matrix C as above
- Decision variables x_{mp} ∈ {0,1}
- Objective: Minimize total cost as above
- Constraints: Each manager and each project assigned exactly once

This is a standard assignment problem formulated as an integer linear program.