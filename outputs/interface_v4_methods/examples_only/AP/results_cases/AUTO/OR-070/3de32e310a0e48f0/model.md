Let:
- M = {1, 2, 3, 4, 5, 6, 7} be the set of managers, corresponding to Manager 1 through Manager 7.
- P = {1, 2, 3, 4, 5, 6, 7} be the set of projects, corresponding to Project 1 through Project 7.
- c_{ij} be the cost for Manager i to complete Project j, as given in the table below.
- x_{ij} ∈ {0, 1} be a binary variable, where x_{ij} = 1 if Manager i is assigned to Project j, 0 otherwise.

Cost matrix C = [c_{ij}]:

|            | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 | Project 6 | Project 7 |
|------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Manager 1  |   2972    |   2727    |   2795    |   2922    |   1302    |   2489    |   1533    |
| Manager 2  |   1094    |   2158    |   2990    |   1844    |   2887    |   2021    |   2288    |
| Manager 3  |   2133    |   1675    |   2422    |   2639    |   1033    |   2261    |   1695    |
| Manager 4  |   1951    |   2309    |   2070    |   2802    |   2328    |   1313    |   2434    |
| Manager 5  |   1269    |   2153    |   1296    |   2685    |   2627    |   1610    |   1641    |
| Manager 6  |   1220    |   1192    |   2907    |   2622    |   2595    |   1261    |   2384    |
| Manager 7  |   1286    |   1659    |   1179    |   1348    |   1420    |   2862    |   1959    |

Mathematical Model:

Decision variables:
x_{ij} = 
    1 if Manager i is assigned to Project j,
    0 otherwise,
for i ∈ M, j ∈ P.

Objective:
Minimize total cost:
\[
\min \sum_{i=1}^{7} \sum_{j=1}^{7} c_{ij} x_{ij}
\]
where c_{ij} is as given in the cost matrix above.

Subject to:
1. Each manager is assigned to exactly one project:
\[
\sum_{j=1}^{7} x_{ij} = 1 \quad \forall i \in M
\]

2. Each project is assigned to exactly one manager:
\[
\sum_{i=1}^{7} x_{ij} = 1 \quad \forall j \in P
\]

3. Binary assignment variables:
\[
x_{ij} \in \{0, 1\} \quad \forall i \in M, j \in P
\]

Where:
- M = {1, 2, 3, 4, 5, 6, 7} (Managers)
- P = {1, 2, 3, 4, 5, 6, 7} (Projects)
- c_{ij} as specified in the table above.

This model ensures each manager is assigned to exactly one project, each project is assigned to exactly one manager, and the total cost is minimized.