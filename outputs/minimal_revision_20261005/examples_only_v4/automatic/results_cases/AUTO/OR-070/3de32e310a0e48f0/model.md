Mathematical Model

Sets:
- Let \( M \) be the set of managers, defined as all unique values in file_0_view_0["Manager"].
- Let \( P \) be the set of projects, defined as all project columns in file_0_view_0: {"Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"}.

Parameters:
- Let \( c_{mp} \) be the cost for manager \( m \in M \) to complete project \( p \in P \), where \( c_{mp} \) is mapped from file_0_view_0 with row index \( m \) (via "Manager") and column \( p \) (project cost column).

Decision Variables:
- \( x_{mp} \in \{0,1\} \) for all \( m \in M, p \in P \), where \( x_{mp} = 1 \) if manager \( m \) is assigned to project \( p \), 0 otherwise.

Objective:
\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} \cdot x_{mp}
\]

Subject to:

1. Each project is assigned to exactly one manager:
\[
\forall p \in P: \quad \sum_{m \in M} x_{mp} = 1
\]

2. Each manager is assigned to at most one project:
\[
\forall m \in M: \quad \sum_{p \in P} x_{mp} \leq 1
\]

3. Binary assignment:
\[
x_{mp} \in \{0,1\} \quad \forall m \in M, p \in P
\]

Data Mapping

- \( M \): file_0_view_0["Manager"]
- \( P \): {"Project 1 Cost", "Project 2 Cost", "Project 3 Cost", "Project 4 Cost", "Project 5 Cost", "Project 6 Cost", "Project 7 Cost"}
- \( c_{mp} \): file_0_view_0, with \( m \) from "Manager" and \( p \) from the corresponding project cost column

This is a classic assignment problem (minimum cost bipartite matching) with all data and indices mapped directly from the provided CSV.