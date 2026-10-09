Let:
- M = {MA, MB, MC, MD, ME, MF} be the set of managers (indexed by i)
- P = {P1, P2, P3, P4, P5, P6} be the set of projects (indexed by j)
- c_{ij} be the cost of assigning manager i to project j, as given below
- x_{ij} ∈ {0,1} be a binary variable: x_{ij} = 1 if manager i is assigned to project j, 0 otherwise

Cost matrix C = [c_{ij}]:

|      | P1   | P2   | P3   | P4   | P5   | P6   |
|------|------|------|------|------|------|------|
| MA   | 2216 | 1911 | 1661 | 2122 | 1442 | 1442 |
| MB   | 1100 | 1271 | 2764 | 2557 | 1036 | 1036 |
| MC   | 2827 | 2784 | 2206 | 2216 | 2677 | 2677 |
| MD   | 2627 | 1273 | 2610 | 1957 | 1594 | 1594 |
| ME   | 3359 | 1003 | 2554 | 1706 | 2065 | 2065 |
| MF   | 1579 | 2289 | 2368 | 1922 | 2740 | 2740 |

Mathematical Model:

Variables:
x_{ij} ∈ {0,1} for all i ∈ M, j ∈ P

Objective:
Minimize ∑_{i ∈ M} ∑_{j ∈ P} c_{ij} x_{ij}
     = 2216 x_{MA,P1} + 1911 x_{MA,P2} + 1661 x_{MA,P3} + 2122 x_{MA,P4} + 1442 x_{MA,P5} + 1442 x_{MA,P6}
     + 1100 x_{MB,P1} + 1271 x_{MB,P2} + 2764 x_{MB,P3} + 2557 x_{MB,P4} + 1036 x_{MB,P5} + 1036 x_{MB,P6}
     + 2827 x_{MC,P1} + 2784 x_{MC,P2} + 2206 x_{MC,P3} + 2216 x_{MC,P4} + 2677 x_{MC,P5} + 2677 x_{MC,P6}
     + 2627 x_{MD,P1} + 1273 x_{MD,P2} + 2610 x_{MD,P3} + 1957 x_{MD,P4} + 1594 x_{MD,P5} + 1594 x_{MD,P6}
     + 3359 x_{ME,P1} + 1003 x_{ME,P2} + 2554 x_{ME,P3} + 1706 x_{ME,P4} + 2065 x_{ME,P5} + 2065 x_{ME,P6}
     + 1579 x_{MF,P1} + 2289 x_{MF,P2} + 2368 x_{MF,P3} + 1922 x_{MF,P4} + 2740 x_{MF,P5} + 2740 x_{MF,P6}

Subject to:
1. Each manager is assigned to exactly one project:
  ∑_{j ∈ P} x_{ij} = 1  for all i ∈ M

2. Each project is assigned to exactly one manager:
  ∑_{i ∈ M} x_{ij} = 1  for all j ∈ P

3. Binary assignment variables:
  x_{ij} ∈ {0,1}  for all i ∈ M, j ∈ P

All parameters (costs, sets, variables) are explicitly listed above. This model ensures a minimum-cost one-to-one assignment of managers to projects, as required.