Mathematical Model

Sets:
- Let 𝑀 be the set of managers, indexed by i, defined as all values in file_0_view_0["Unnamed: 0"].
- Let 𝑃 be the set of projects, indexed by j, defined as all columns in file_0_view_0 except "Unnamed: 0".

Parameters:
- Let c_{i,j} be the cost of assigning manager i to project j, given by file_0_view_0 with mapping: c_{i,j} = file_0_view_0[manager=i, project=j].

Decision Variables:
- x_{i,j} ∈ {0,1}, where x_{i,j} = 1 if manager i is assigned to project j, 0 otherwise, for all i ∈ 𝑀, j ∈ 𝑃.

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

Data Mapping

- Set of managers 𝑀: All values in file_0_view_0["Unnamed: 0"]
- Set of projects 𝑃: All columns in file_0_view_0 except "Unnamed: 0"
- Cost parameter c_{i,j}: file_0_view_0, with i matched to "Unnamed: 0" and j matched to project column name

All indices, parameters, and constraints are mapped directly to the data in table_id file_0_view_0.