Let:
- S = {S1, S2, S3, S4} be the set of production plants.
- C = {C1, C2, C3, C4} be the set of retail outlets.
- x_{i,j} = number of beverage units shipped from plant i ∈ S to customer j ∈ C.

Parameters:
- Demand for each customer:
    - d_{C1} = 94
    - d_{C2} = 39
    - d_{C3} = 65
    - d_{C4} = 435
- Supply capacity for each plant:
    - s_{S1} = 2531
    - s_{S2} = 20
    - s_{S3} = 210
    - s_{S4} = 241
- Transportation costs per unit (c_{i,j}):
    - From S1: c_{S1,C1} = 543.756480860856, c_{S1,C2} = 23.685276141764653, c_{S1,C3} = 23.676386730773032, c_{S1,C4} = 447.75143678673766
    - From S2: c_{S2,C1} = 883.9151090405642, c_{S2,C2} = 0.04977684765576961, c_{S2,C3} = 0.0350986687216299, c_{S2,C4} = 44.45588531711622
    - From S3: c_{S3,C1} = 537.3456896658107, c_{S3,C2} = 23.769274659075112, c_{S3,C3} = 498.95659249465467, c_{S3,C4} = 440.60737890439776
    - From S4: c_{S4,C1} = 1791.493192397229, c_{S4,C2} = 68.21633865655126, c_{S4,C3} = 1432.4837339656747, c_{S4,C4} = 1527.7635425462734

Variables:
- x_{i,j} ≥ 0, ∀ i ∈ S, j ∈ C

Model:

Minimize total transportation cost:
\[
\text{Minimize} \quad
543.756480860856\,x_{S1,C1} + 23.685276141764653\,x_{S1,C2} + 23.676386730773032\,x_{S1,C3} + 447.75143678673766\,x_{S1,C4} \\
+ 883.9151090405642\,x_{S2,C1} + 0.04977684765576961\,x_{S2,C2} + 0.0350986687216299\,x_{S2,C3} + 44.45588531711622\,x_{S2,C4} \\
+ 537.3456896658107\,x_{S3,C1} + 23.769274659075112\,x_{S3,C2} + 498.95659249465467\,x_{S3,C3} + 440.60737890439776\,x_{S3,C4} \\
+ 1791.493192397229\,x_{S4,C1} + 68.21633865655126\,x_{S4,C2} + 1432.4837339656747\,x_{S4,C3} + 1527.7635425462734\,x_{S4,C4}
\]

Subject to:

1. Demand satisfaction for each customer:
\[
x_{S1,Cj} + x_{S2,Cj} + x_{S3,Cj} + x_{S4,Cj} = d_{Cj}, \quad \forall j \in \{C1, C2, C3, C4\}
\]
That is,
\[
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} = 94 \\
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} = 39 \\
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} = 65 \\
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} = 435
\]

2. Supply capacity for each plant:
\[
x_{Si,C1} + x_{Si,C2} + x_{Si,C3} + x_{Si,C4} \leq s_{Si}, \quad \forall i \in \{S1, S2, S3, S4\}
\]
That is,
\[
x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} \leq 2531 \\
x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} \leq 20 \\
x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} \leq 210 \\
x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} \leq 241
\]

3. Non-negativity:
\[
x_{i,j} \geq 0, \quad \forall i \in S, j \in C
\]

This is a complete numerical linear programming formulation for BrewCo's transportation problem.