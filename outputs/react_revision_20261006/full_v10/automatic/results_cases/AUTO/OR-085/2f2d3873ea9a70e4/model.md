Mathematical Model (Traveling Salesman Problem)

Sets:
N = {1, 2, ..., 15}  // set of locations (customers), with 1 as the depot (start/end)

Parameters:
d_{ij} = distance from location i to location j, for all i, j ∈ N, i ≠ j
  (Data Mapping: d_{ij} is the entry in 20.csv, table_id: file_0_view_0, with i as the row index (1-based), j as the column header (2–15), and d_{ii} = 0 for all i)

Decision Variables:
x_{ij} ∈ {0,1}  for all i, j ∈ N, i ≠ j
  x_{ij} = 1 if the route goes directly from i to j, 0 otherwise

u_i ∈ [2, 15]  for all i ∈ N, i ≠ 1
  (subtour elimination variables; u_1 = 1)

Objective:
Minimize  ∑_{i∈N} ∑_{j∈N, j≠i} d_{ij} x_{ij}

Subject to:
1. Leave each location exactly once:
  ∑_{j∈N, j≠i} x_{ij} = 1  for all i ∈ N

2. Enter each location exactly once:
  ∑_{i∈N, i≠j} x_{ij} = 1  for all j ∈ N

3. Subtour elimination (Miller-Tucker-Zemlin constraints):
  u_i - u_j + 15 x_{ij} ≤ 14  for all i, j ∈ N, i ≠ j, i ≠ 1, j ≠ 1

4. Start and end at location 1:
  The tour starts and ends at node 1 (enforced by constraints 1 and 2).

5. Variable domains:
  x_{ij} ∈ {0,1}  for all i, j ∈ N, i ≠ j
  u_1 = 1; u_i ∈ [2, 15] for i ∈ N, i ≠ 1

Data Mapping:
- d_{ij}: file_0_view_0, 20.csv, with i = source_row+1, j = column header (2–15), symmetric, d_{ii} = 0
- N = {1,2,...,15} (locations/customers)
- All variables and constraints as above

This model finds the shortest possible route starting and ending at location 1, visiting each of the 15 locations exactly once.