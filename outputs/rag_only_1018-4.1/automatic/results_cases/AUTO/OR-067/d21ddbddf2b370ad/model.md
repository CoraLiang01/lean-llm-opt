Let:
- M = {MA, MB, MC} be the set of managers, indexed by i ∈ {1,2,3}
- P = {P1, P2, P3} be the set of projects, indexed by j ∈ {1,2,3}
- Let x[i,j] ∈ {0,1} be a binary variable: x[i,j] = 1 if manager i is assigned to project j, 0 otherwise.

Cost matrix C (rows: managers, columns: projects):

C = 
|      | P1   | P2   | P3   |
|------|------|------|------|
| MA   | 3000 | 3200 | 3100 |
| MB   | 2800 | 3300 | 2900 |
| MC   | 2900 | 3100 | 3000 |

Explicitly, as a matrix:
C = 
⎡ 3000  3200  3100 ⎤
⎢ 2800  3300  2900 ⎥
⎣ 2900  3100  3000 ⎦

Eligibility mask:
All pairs are eligible (all entries have a cost), so the eligibility mask is a 3x3 matrix of all ones:
E = 
⎡ 1  1  1 ⎤
⎢ 1  1  1 ⎥
⎣ 1  1  1 ⎦

Mathematical Model:

Variables:
x[i,j] ∈ {0,1} for i ∈ {1,2,3}, j ∈ {1,2,3}

Objective:
Minimize total cost:
minimize   ∑_{i=1}^3 ∑_{j=1}^3 C[i,j] * x[i,j]
= 3000 x[1,1] + 3200 x[1,2] + 3100 x[1,3]
  + 2800 x[2,1] + 3300 x[2,2] + 2900 x[2,3]
  + 2900 x[3,1] + 3100 x[3,2] + 3000 x[3,3]

Subject to:
1. Each manager is assigned to exactly one project:
   ∑_{j=1}^3 x[i,j] = 1   for all i ∈ {1,2,3}

2. Each project is assigned to exactly one manager:
   ∑_{i=1}^3 x[i,j] = 1   for all j ∈ {1,2,3}

3. Binary variables:
   x[i,j] ∈ {0,1}   for all i, j

All pairs are eligible, so no x[i,j] == 0 constraints are needed.

Summary of required parameters:
- Cost matrix C as above
- Eligibility mask E as above (all ones)
- Decision variables x[i,j] ∈ {0,1}
- Objective and constraints as stated

This is a standard linear assignment problem (minimum cost bipartite matching) with all pairs eligible.