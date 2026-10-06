Mathematical Model

Sets:
- Let 𝑀 be the set of managers, defined as all values in file_0_view_0.Manager.
- Let 𝑃 be the set of projects, defined as all project columns in file_0_view_0: {"Project 1 Cost", "Project 2 Cost", ..., "Project 11 Cost"}.

Parameters:
- For each manager m ∈ 𝑀 and project p ∈ 𝑃, let c_{m,p} = value in file_0_view_0 where Manager = m and column = p.

Decision Variables:
- For each manager m ∈ 𝑀 and project p ∈ 𝑃:
  x_{m,p} ∈ {0,1}, where x_{m,p} = 1 if manager m is assigned to project p, 0 otherwise.

Objective:
Minimize total assignment cost:
\[
\min \sum_{m \in M} \sum_{p \in P} c_{m,p} \cdot x_{m,p}
\]

Subject to:

1. Each manager is assigned to exactly one project:
\[
\forall m \in M: \quad \sum_{p \in P} x_{m,p} = 1
\]

2. Each project is assigned to exactly one manager:
\[
\forall p \in P: \quad \sum_{m \in M} x_{m,p} = 1
\]

3. Binary assignment variables:
\[
x_{m,p} \in \{0,1\} \quad \forall m \in M,\, p \in P
\]

Data Mapping

- Set of managers M: file_0_view_0.Manager
- Set of projects P: {all columns in file_0_view_0 with names matching "Project * Cost"}
- Cost parameter c_{m,p}: file_0_view_0, row where Manager = m, column = p
- Decision variable x_{m,p}: assignment of manager m to project p

All indices, parameters, and constraints are mapped directly to the data in file_0_view_0 from manager_project_costs.csv.