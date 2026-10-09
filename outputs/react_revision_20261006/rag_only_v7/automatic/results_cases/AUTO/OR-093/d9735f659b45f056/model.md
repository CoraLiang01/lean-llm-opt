Mathematical Model:

Sets:
- Let M be the set of machines, indexed by i. (M = {"M1", "M2", ..., "M12"})
- Let T be the set of tasks, indexed by j. (T = {"A", "B", ..., "L"})

Parameters:
- c_{ij}: Cost of assigning machine i ∈ M to task j ∈ T.

Decision Variables:
- x_{ij} ∈ {0,1}: 1 if machine i is assigned to task j, 0 otherwise.

Objective:
Minimize total assignment cost:
\[
\min \sum_{i \in M} \sum_{j \in T} c_{ij} x_{ij}
\]

Subject to:
1. Each machine is assigned to exactly one task:
\[
\sum_{j \in T} x_{ij} = 1 \quad \forall i \in M
\]
2. Each task is assigned to exactly one machine:
\[
\sum_{i \in M} x_{ij} = 1 \quad \forall j \in T
\]
3. Binary assignment variables:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in T
\]

Data Mapping:
- Set M (machines): file_0_view_0, column "Machine"
- Set T (tasks): file_0_view_0, columns {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"}
- Parameter c_{ij}: file_0_view_0, value at row with "Machine" = i and column = j

All indices, parameters, and constraints are defined exactly as mapped from the current CSV data.