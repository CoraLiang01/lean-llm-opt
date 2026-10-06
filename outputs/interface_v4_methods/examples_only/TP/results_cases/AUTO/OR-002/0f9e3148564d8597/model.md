Let:
- S = {S1, S2, ..., S11} be the set of Walmart stores (supply nodes).
- C = {C1, C2, ..., C12} be the set of customer groups (demand nodes).
- x_{i,j} = quantity transported from store i ∈ S to customer group j ∈ C.

Parameters (from CSVs, source order preserved):

Customer Demands (customer_demand.csv):
- demand_C1 = 11
- demand_C2 = 1148
- demand_C3 = 54
- demand_C4 = 833
- demand_C5 = 154
- demand_C6 = 551
- demand_C7 = 7081
- demand_C8 = 76
- demand_C9 = 66
- demand_C10 = 174
- demand_C11 = 15
- demand_C12 = 680

Supply Capacities (supply_capacity.csv):
- supply_S1 = 4
- supply_S2 = 575
- supply_S3 = 1504
- supply_S4 = 178
- supply_S5 = 228
- supply_S6 = 50
- supply_S7 = 3
- supply_S8 = 6148
- supply_S9 = 6
- supply_S10 = 10673
- supply_S11 = 174

Transportation Costs (transportation_costs.csv): Let c_{i,j} denote the cost per unit from store i to customer group j, as given in the CSV (not shown here, but to be filled in source order).

Variables:
- x_{i,j} ≥ 0, ∀ i ∈ S, j ∈ C

Objective:
Minimize total transportation cost:
minimize ∑_{i ∈ S} ∑_{j ∈ C} c_{i,j} x_{i,j}

Subject to:

1. Demand satisfaction for each customer group:
  ∑_{i ∈ S} x_{i,j} = demand_Cj  ∀ j ∈ C

2. Supply capacity for each Walmart store:
  ∑_{j ∈ C} x_{i,j} ≤ supply_Si  ∀ i ∈ S

3. Non-negativity:
  x_{i,j} ≥ 0  ∀ i ∈ S, j ∈ C

Where all identifiers and coefficients (demands, supplies, costs) are as listed above and as provided in the CSVs, preserving source order. The objective sense is minimization, and all variables are continuous and non-negative.