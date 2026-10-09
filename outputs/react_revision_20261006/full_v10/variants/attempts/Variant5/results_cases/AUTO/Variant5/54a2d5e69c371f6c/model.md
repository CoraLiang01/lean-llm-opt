Mathematical Model

Sets:
I = {M1, M2, M3, M4}         // Teams (from machine_capacity.csv)
J = {J1, J2, J3, J4, J5, J6, J7, J8}   // Jobs (from assignment_costs.csv and assignment_resources.csv)

Parameters:
c_ij = assignment cost of assigning job j to team i (from assignment_costs.csv, table_id: file_1_view_0, columns: Machine, J1–J8)
a_ij = capacity consumed on team i by assigning job j (from assignment_resources.csv, table_id: file_2_view_0, columns: Machine, J1–J8)
Cap_i = available capacity of team i (from machine_capacity.csv, table_id: file_0_view_0, columns: Machine, Capacity)

Decision Variables:
x_ij ∈ {0,1}   for all i ∈ I, j ∈ J
  x_ij = 1 if job j is assigned to team i, 0 otherwise

Objective:
Minimize total assignment cost:
minimize ∑_{i∈I} ∑_{j∈J} c_ij x_ij

Subject to:
1. Each job is assigned to exactly one team:
  for all j ∈ J:
   ∑_{i∈I} x_ij = 1

2. Team capacity constraints:
  for all i ∈ I:
   ∑_{j∈J} a_ij x_ij ≤ Cap_i

3. Binary assignment variables:
  for all i ∈ I, j ∈ J:
   x_ij ∈ {0,1}

Data Mapping:
- Teams I: Machine from machine_capacity.csv (file_0_view_0)
- Jobs J: J1–J8 from assignment_costs.csv and assignment_resources.csv (file_1_view_0, file_2_view_0)
- c_ij: assignment_costs.csv, table_id: file_1_view_0, columns Machine, J1–J8
- a_ij: assignment_resources.csv, table_id: file_2_view_0, columns Machine, J1–J8
- Cap_i: machine_capacity.csv, table_id: file_0_view_0, columns Machine, Capacity

All sets, parameters, and variables are defined exactly as in the source data.