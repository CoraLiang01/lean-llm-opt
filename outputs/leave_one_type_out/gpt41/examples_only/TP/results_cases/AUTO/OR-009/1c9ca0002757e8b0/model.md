Let:
- S = {S1, S2, S3, S4} be the set of production plants.
- C = {C1, C2, C3, C4} be the set of retail outlets.
- x_{s,c} = quantity shipped from plant s ∈ S to outlet c ∈ C (decision variables, x_{s,c} ≥ 0).

Parameters (from CSVs, source order preserved):

Customer demand (customer_demand.csv):
- demand_C1 = 94
- demand_C2 = 39
- demand_C3 = 65
- demand_C4 = 435

Supply capacity (supply_capacity.csv):
- supply_S1 = 2531
- supply_S2 = 20
- supply_S3 = 210
- supply_S4 = 241

Transportation costs (transportation_costs.csv):

|         | C1                | C2                | C3                | C4                |
|---------|-------------------|-------------------|-------------------|-------------------|
| S1      | 543.756480860856  | 23.685276141764653| 23.676386730773032| 447.75143678673766|
| S2      | 883.9151090405642 | 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
| S3      | 537.3456896658107 | 23.769274659075112| 498.95659249465467| 440.60737890439776|
| S4      | 1791.493192397229 | 68.21633865655126 | 1432.4837339656747| 1527.7635425462734|

Variables:
x_{S1,C1}, x_{S1,C2}, x_{S1,C3}, x_{S1,C4} ≥ 0  
x_{S2,C1}, x_{S2,C2}, x_{S2,C3}, x_{S2,C4} ≥ 0  
x_{S3,C1}, x_{S3,C2}, x_{S3,C3}, x_{S3,C4} ≥ 0  
x_{S4,C1}, x_{S4,C2}, x_{S4,C3}, x_{S4,C4} ≥ 0  

Objective (minimize total transportation cost, source order preserved):

Minimize  
543.756480860856 x_{S1,C1} + 23.685276141764653 x_{S1,C2} + 23.676386730773032 x_{S1,C3} + 447.75143678673766 x_{S1,C4}  
+ 883.9151090405642 x_{S2,C1} + 0.04977684765576961 x_{S2,C2} + 0.0350986687216299 x_{S2,C3} + 44.45588531711622 x_{S2,C4}  
+ 537.3456896658107 x_{S3,C1} + 23.769274659075112 x_{S3,C2} + 498.95659249465467 x_{S3,C3} + 440.60737890439776 x_{S3,C4}  
+ 1791.493192397229 x_{S4,C1} + 68.21633865655126 x_{S4,C2} + 1432.4837339656747 x_{S4,C3} + 1527.7635425462734 x_{S4,C4}

Subject to:

Demand constraints (each outlet’s demand must be met):
x_{S1,C1} + x_{S2,C1} + x_{S3,C1} + x_{S4,C1} = 94  
x_{S1,C2} + x_{S2,C2} + x_{S3,C2} + x_{S4,C2} = 39  
x_{S1,C3} + x_{S2,C3} + x_{S3,C3} + x_{S4,C3} = 65  
x_{S1,C4} + x_{S2,C4} + x_{S3,C4} + x_{S4,C4} = 435  

Supply constraints (each plant’s total shipments cannot exceed its capacity):
x_{S1,C1} + x_{S1,C2} + x_{S1,C3} + x_{S1,C4} ≤ 2531  
x_{S2,C1} + x_{S2,C2} + x_{S2,C3} + x_{S2,C4} ≤ 20  
x_{S3,C1} + x_{S3,C2} + x_{S3,C3} + x_{S3,C4} ≤ 210  
x_{S4,C1} + x_{S4,C2} + x_{S4,C3} + x_{S4,C4} ≤ 241  

Non-negativity:
x_{s,c} ≥ 0 for all s ∈ S, c ∈ C

This is the complete numerical formulation for BrewCo’s transportation optimization problem, with all identifiers and coefficients preserved in source order.