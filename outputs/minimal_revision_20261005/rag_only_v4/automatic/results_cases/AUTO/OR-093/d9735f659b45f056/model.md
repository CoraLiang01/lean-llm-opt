Mathematical Model

Sets:
- Let 𝑀 be the set of machines, indexed by i, where 𝑀 = {M1, M2, ..., M12} (from file_0_view_0, column "Machine").
- Let 𝑇 be the set of tasks, indexed by j, where 𝑇 = {A, B, ..., L} (from file_0_view_0, columns "A" through "L").

Parameters:
- Let c_{ij} be the machining cost of assigning machine i ∈ 𝑀 to task j ∈ 𝑇, as given in file_0_view_0, with i from "Machine" and j from columns "A" through "L".

Decision Variables:
- x_{ij} ∈ {0,1} for all i ∈ 𝑀, j ∈ 𝑇
  x_{ij} = 1 if machine i is assigned to task j, 0 otherwise.

Objective:
Minimize total assignment cost:
\[
\min \sum_{i \in M} \sum_{j \in T} c_{ij} \cdot x_{ij}
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

Data Mapping

- Set 𝑀: file_0_view_0, column "Machine"
- Set 𝑇: file_0_view_0, columns "A" through "L"
- Parameter c_{ij}: file_0_view_0, value at row with "Machine" = i and column = j
- Decision variable x_{ij}: assignment of machine i ∈ 𝑀 to task j ∈ 𝑇

This is a classical assignment problem (minimum-cost bipartite matching) with all data and indices mapped directly from file_0_view_0.