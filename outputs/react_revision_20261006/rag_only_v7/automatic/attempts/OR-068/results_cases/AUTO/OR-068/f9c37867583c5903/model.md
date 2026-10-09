Mathematical Model (Assignment Problem):

Sets:
- Let M be the set of managers, indexed by i. (M = {"MA", "MB", "MC", "MD", "ME", "MF"} from file_0_view_0, column "Unnamed: 0")
- Let P be the set of projects, indexed by j. (P = {"P1", "P2", "P3", "P4", "P5", "P6"} from file_0_view_0, columns)

Parameters:
- Let c_{i,j} be the cost of assigning manager i to project j, as given in file_0_view_0, column j, row with "Unnamed: 0" = i.

Decision Variables:
- x_{i,j} ∈ {0,1}, for all i ∈ M, j ∈ P
    - x_{i,j} = 1 if manager i is assigned to project j, 0 otherwise.

Objective:
Minimize total assignment cost:
\[
\min \sum_{i \in M} \sum_{j \in P} c_{i,j} \cdot x_{i,j}
\]

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j \in P} x_{i,j} = 1 \quad \forall i \in M
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i \in M} x_{i,j} = 1 \quad \forall j \in P
\]

3. Binary assignment variables:
\[
x_{i,j} \in \{0,1\} \quad \forall i \in M, j \in P
\]

Data Mapping:
- M (managers): file_0_view_0, column "Unnamed: 0"
- P (projects): file_0_view_0, columns {"P1", "P2", "P3", "P4", "P5", "P6"}
- c_{i,j}: file_0_view_0, value at row with "Unnamed: 0" = i, column = j

This model finds the minimum-cost one-to-one assignment of managers to projects using the cost matrix from the provided CSV.