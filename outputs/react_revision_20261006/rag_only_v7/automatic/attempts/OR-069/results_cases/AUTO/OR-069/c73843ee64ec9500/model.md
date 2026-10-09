Mathematical Model:

Sets:
- Let \( M \) be the set of managers, indexed by \( i \), corresponding to all "Manager" entries in table_id file_0_view_0.
- Let \( P \) be the set of projects, indexed by \( j \), corresponding to all project columns in table_id file_0_view_0 (i.e., columns with names "Project k Cost" for \( k = 1, \ldots, 11 \)).

Parameters:
- Let \( c_{ij} \) be the cost of assigning manager \( i \) to project \( j \), given by the value in table_id file_0_view_0, row "Manager" = \( i \), column "Project k Cost" corresponding to project \( j \).

Decision Variables:
- \( x_{ij} \in \{0,1\} \) for all \( i \in M, j \in P \): \( x_{ij} = 1 \) if manager \( i \) is assigned to project \( j \), 0 otherwise.

Objective:
\[
\min \sum_{i \in M} \sum_{j \in P} c_{ij} x_{ij}
\]

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

Data Mapping:
- \( M \): All values in column "Manager" of table_id file_0_view_0.
- \( P \): All project columns in table_id file_0_view_0, i.e., columns with names "Project 1 Cost", ..., "Project 11 Cost".
- \( c_{ij} \): Value at row where "Manager" = \( i \), column "Project k Cost" for project \( j \), in table_id file_0_view_0.

Index sets, parameters, and all constraints are defined exactly by the current data in manager_project_costs.csv (table_id file_0_view_0).