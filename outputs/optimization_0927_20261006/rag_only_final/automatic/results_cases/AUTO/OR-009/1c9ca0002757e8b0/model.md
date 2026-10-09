Let S = {S1, S2, S3, S4} be the set of production plants, and C = {C1, C2, C3, C4} be the set of retail outlets.

Decision Variables:
Let x_{i,j} ≥ 0 denote the number of beverage units shipped from plant i ∈ S to customer j ∈ C.

Objective:
Minimize total transportation cost:
minimize
543.756480860856·x_{S1,C1} + 23.685276141764653·x_{S1,C2} + 23.676386730773032·x_{S1,C3} + 447.75143678673766·x_{S1,C4}
+ 883.9151090405642·x_{S2,C1} + 0.04977684765576961·x_{S2,C2} + 0.0350986687216299·x_{S2,C3} + 44.45588531711622·x_{S2,C4}
+ 537.3456896658107·x_{S3,C1} + 23.769274659075112·x_{S3,C2} + 498.95659249465467·x_{S3,C3} + 440.60737890439776·x_{S3,C4}
+ 1791.493192397229·x_{S4,C1} + 68.21633865655126·x_{S4,C2} + 1432.4837339656747·x_{S4,C3} + 1527.7635425462734·x_{S4,C4}

Subject to:

Demand satisfaction (for each customer):
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} = 94
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} = 39
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} = 65
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} = 435

Supply capacity (for each plant):
x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} ≤ 2531
x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} ≤ 20
x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} ≤ 210
x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} ≤ 241

Nonnegativity:
x_{i,j} ≥ 0 for all i ∈ S, j ∈ C

All coefficients, identifiers, and constraints are preserved in source order as required.