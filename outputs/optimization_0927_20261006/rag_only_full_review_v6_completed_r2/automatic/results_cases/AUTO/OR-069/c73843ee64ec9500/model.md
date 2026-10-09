Mathematical Model for Minimum-Cost One-to-One Assignment of Managers to Projects

Sets:
- M = {1, 2, ..., 11} (Managers: Manager 1, ..., Manager 11)
- P = {1, 2, ..., 11} (Projects: Project 1, ..., Project 11)

Parameters:
- c_ij: Cost of assigning manager i to project j, as given in the table above.

Decision Variables:
- x_ij ∈ {0,1}, for all i ∈ M, j ∈ P

Objective:
\[
\min \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j=1}^{11} x_{ij} = 1 \quad \forall i = 1, ..., 11
\]
2. Each project is assigned to exactly one manager:
\[
\sum_{i=1}^{11} x_{ij} = 1 \quad \forall j = 1, ..., 11
\]
3. Binary assignment variables:
\[
x_{ij} \in \{0,1\} \quad \forall i = 1, ..., 11; \; j = 1, ..., 11
\]

Where the cost matrix [c_ij] is as specified above.