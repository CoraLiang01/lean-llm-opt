Sets:
- S = {S1, S2, S3, S4} (production plants)
- C = {C1, C2, C3, C4} (customers/retail outlets)

Parameters:
- demand_C1 = 94
- demand_C2 = 39
- demand_C3 = 65
- demand_C4 = 435

- supply_capacity_S1 = 2531
- supply_capacity_S2 = 20
- supply_capacity_S3 = 210
- supply_capacity_S4 = 241

Transportation costs (per unit shipped from plant Si to customer Cj):
- cost_S1_C1 = 543.756480860856
- cost_S1_C2 = 23.685276141764653
- cost_S1_C3 = 23.676386730773032
- cost_S1_C4 = 447.75143678673766

- cost_S2_C1 = 883.9151090405642
- cost_S2_C2 = 0.04977684765576961
- cost_S2_C3 = 0.0350986687216299
- cost_S2_C4 = 44.45588531711622

- cost_S3_C1 = 537.3456896658107
- cost_S3_C2 = 23.769274659075112
- cost_S3_C3 = 498.95659249465467
- cost_S3_C4 = 440.60737890439776

- cost_S4_C1 = 1791.493192397229
- cost_S4_C2 = 68.21633865655126
- cost_S4_C3 = 1432.4837339656747
- cost_S4_C4 = 1527.7635425462734

Decision Variables:
- x_{i,j} = quantity of beverages shipped from plant i ∈ S to customer j ∈ C
  (x_{i,j} ≥ 0, continuous)

Objective:
Minimize total transportation cost:
minimize
  543.756480860856 x_{S1,C1} + 23.685276141764653 x_{S1,C2} + 23.676386730773032 x_{S1,C3} + 447.75143678673766 x_{S1,C4}
+ 883.9151090405642 x_{S2,C1} + 0.04977684765576961 x_{S2,C2} + 0.0350986687216299 x_{S2,C3} + 44.45588531711622 x_{S2,C4}
+ 537.3456896658107 x_{S3,C1} + 23.769274659075112 x_{S3,C2} + 498.95659249465467 x_{S3,C3} + 440.60737890439776 x_{S3,C4}
+ 1791.493192397229 x_{S4,C1} + 68.21633865655126 x_{S4,C2} + 1432.4837339656747 x_{S4,C3} + 1527.7635425462734 x_{S4,C4}

Subject to:

1. Demand satisfaction for each customer:
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} = 94
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} = 39
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} = 65
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} = 435

2. Supply capacity for each plant:
x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} ≤ 2531
x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} ≤ 20
x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} ≤ 210
x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} ≤ 241

3. Non-negativity:
x_{i,j} ≥ 0 for all i ∈ S, j ∈ C

This is the complete numerical formulation of BrewCo's transportation optimization problem, using all identifiers and coefficients as retrieved from the source data.