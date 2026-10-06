Let x_{ij} denote the number of beverage units shipped from plant i to customer j, where i ∈ {S1, S2, S3, S4} and j ∈ {C1, C2, C3, C4}.

Parameters (from CSV evidence):

Customer Demands (customer_demand.csv):
- C1: 94
- C2: 39
- C3: 65
- C4: 435

Plant Capacities (supply_capacity.csv):
- S1: 2531
- S2: 20
- S3: 210
- S4: 241

Transportation Costs per Unit (transportation_costs.csv):

|      | C1           | C2           | C3           | C4           |
|------|--------------|--------------|--------------|--------------|
| S1   | 543.7564809  | 23.68527614  | 23.67638673  | 447.7514368  |
| S2   | 883.9151090  | 0.04977685   | 0.03509867   | 44.45588532  |
| S3   | 537.3456897  | 23.76927466  | 498.9565925  | 440.6073789  |
| S4   | 1791.493192  | 68.21633866  | 1432.483734  | 1527.763543  |

Mathematical Optimization Model:

Decision Variables:
x_{ij} ≥ 0, continuous, for all i ∈ {S1, S2, S3, S4}, j ∈ {C1, C2, C3, C4}

Objective:
Minimize total transportation cost:
minimize
543.756480860856 x_{S1,C1} + 23.685276141764653 x_{S1,C2} + 23.676386730773032 x_{S1,C3} + 447.75143678673766 x_{S1,C4}
+ 883.9151090405642 x_{S2,C1} + 0.04977684765576961 x_{S2,C2} + 0.0350986687216299 x_{S2,C3} + 44.45588531711622 x_{S2,C4}
+ 537.3456896658107 x_{S3,C1} + 23.769274659075112 x_{S3,C2} + 498.95659249465467 x_{S3,C3} + 440.60737890439776 x_{S3,C4}
+ 1791.493192397229 x_{S4,C1} + 68.21633865655126 x_{S4,C2} + 1432.4837339656747 x_{S4,C3} + 1527.7635425462734 x_{S4,C4}

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

Non-negativity:
x_{ij} ≥ 0 for all i, j

This is the complete numerical formulation of BrewCo's transportation optimization problem, using all coefficients and identifiers as provided in the CSV evidence.