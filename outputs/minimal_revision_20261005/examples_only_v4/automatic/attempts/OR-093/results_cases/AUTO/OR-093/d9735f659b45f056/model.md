Mathematical Model

Sets:
- Let \( M \) be the set of machines, indexed by \( i \), where \( M = \{\text{M1}, \text{M2}, \ldots, \text{M12}\} \) (from file_0_view_0, column "Machine").
- Let \( T \) be the set of tasks, indexed by \( j \), where \( T = \{\text{A}, \text{B}, \ldots, \text{L}\} \) (from file_0_view_0, columns "A" through "L").

Parameters:
- Let \( c_{ij} \) be the cost of assigning machine \( i \) to task \( j \), where \( c_{ij} \) is given by the entry in file_0_view_0, row with "Machine" = \( i \), column \( j \).

Decision Variables:
- \( x_{ij} \in \{0,1\} \) for all \( i \in M, j \in T \), where \( x_{ij} = 1 \) if machine \( i \) is assigned to task \( j \), 0 otherwise.

Objective:
\[
\min \sum_{i \in M} \sum_{j \in T} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{j \in T} x_{ij} = 1 \quad \forall i \in M \qquad \text{(each machine assigned to exactly one task)}
\]
\[
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in T \qquad \text{(each task assigned to exactly one machine)}
\]
\[
x_{ij} \in \{0,1\} \quad \forall i \in M, j \in T
\]

Data Mapping:
- \( M \): All values in file_0_view_0, column "Machine"
- \( T \): All column names in file_0_view_0, columns "A" through "L"
- \( c_{ij} \): file_0_view_0, entry at row where "Machine" = \( i \), column \( j \)