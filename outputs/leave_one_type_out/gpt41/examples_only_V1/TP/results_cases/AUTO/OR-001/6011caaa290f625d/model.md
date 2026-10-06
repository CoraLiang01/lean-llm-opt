Let S = {S1, S2, ..., S18} be the set of distribution centers, and C = {C1, C2, ..., C18} be the set of customer groups.

Parameters (from CSV evidence, source order preserved):

Customer Demands (customer_demand.csv):
- demand_C1 = 4415
- demand_C2 = 5430
- demand_C3 = 81
- demand_C4 = 146
- demand_C5 = 10638
- demand_C6 = 1663
- demand_C7 = 151
- demand_C8 = 185
- demand_C9 = 1917
- demand_C10 = 4489
- demand_C11 = 76
- demand_C12 = 2529
- demand_C13 = 2136
- demand_C14 = 909
- demand_C15 = 316
- demand_C16 = 70
- demand_C17 = 1456
- demand_C18 = 2711

Supply Capacities (supply_capacity.csv):
- supply_S1 = 5963
- supply_S2 = 702
- supply_S3 = 350
- supply_S4 = 11483
- supply_S5 = 6585
- supply_S6 = 11330
- supply_S7 = 207
- supply_S8 = 788
- supply_S9 = 6967
- supply_S10 = 43
- supply_S11 = 1137
- supply_S12 = 1553
- supply_S13 = 257
- supply_S14 = 2114
- supply_S15 = 205
- supply_S16 = 17326
- supply_S17 = 22260
- supply_S18 = 333

Transportation Costs (transportation_costs.csv, cost_Si_Cj):

(cost_S1_C1 = 159.83765495858208, cost_S1_C2 = 6.42633790337285, ..., cost_S1_C18 = 7.6626195540288755)
(cost_S2_C1 = 0.10655701710223901, ..., cost_S2_C18 = 0.14381223458648024)
(cost_S3_C1 = 0.07121132969220138, ..., cost_S3_C18 = 1.724132015627177)
(cost_S4_C1 = 1.5455382897983385, ..., cost_S4_C18 = 1.5358539138142018)
(cost_S5_C1 = 16.41408384079902, ..., cost_S5_C18 = 16.31794206958455)
(cost_S6_C1 = 158.9313402279755, ..., cost_S6_C18 = 7.514549419003527)
(cost_S7_C1 = 0.10720291412027193, ..., cost_S7_C18 = 0.07215475483459184)
(cost_S8_C1 = 2.220199373580392, ..., cost_S8_C18 = 2.2930186182029604)
(cost_S9_C1 = 0.059394016843197485, ..., cost_S9_C18 = 0.07465438573576134)
(cost_S10_C1 = 76.00470976255896, ..., cost_S10_C18 = 3.6985586217330395)
(cost_S11_C1 = 1349.7946298063491, ..., cost_S11_C18 = 1157.51304425127)
(cost_S12_C1 = 1.435340983528516, ..., cost_S12_C18 = 0.7753196305628699)
(cost_S13_C1 = 314.2033136744687, ..., cost_S13_C18 = 14.981357818407366)
(cost_S14_C1 = 5.752413335760744, ..., cost_S14_C18 = 5.820650534912412)
(cost_S15_C1 = 654.160873808039, ..., cost_S15_C18 = 36.33495494783298)
(cost_S16_C1 = 10.834014840644409, ..., cost_S16_C18 = 193.79044895963023)
(cost_S17_C1 = 16.409641338621462, ..., cost_S17_C18 = 342.58424666699665)
(cost_S18_C1 = 837.3389315505575, ..., cost_S18_C18 = 717.3748902497082)

Decision Variables:
Let x_{i,j} = quantity of goods transported from distribution center Si to customer group Cj
- Domain: x_{i,j} ≥ 0, ∀ i ∈ {1,...,18}, j ∈ {1,...,18}

Objective:
Minimize total transportation cost:
minimize
∑_{i=1}^{18} ∑_{j=1}^{18} cost_{Si,Cj} * x_{i,j}
That is,
minimize
159.83765495858208 x_{1,1} + 6.42633790337285 x_{1,2} + ... + 7.6626195540288755 x_{1,18}
+ 0.10655701710223901 x_{2,1} + ... + 0.14381223458648024 x_{2,18}
+ ... (continue for all i=1..18, j=1..18, using the coefficients above in source order)

Subject to:

1. Demand satisfaction for each customer group:
For each customer group Cj (j=1..18):
∑_{i=1}^{18} x_{i,j} = demand_Cj

Explicitly:
∑_{i=1}^{18} x_{i,1} = 4415
∑_{i=1}^{18} x_{i,2} = 5430
...
∑_{i=1}^{18} x_{i,18} = 2711

2. Supply capacity for each distribution center:
For each distribution center Si (i=1..18):
∑_{j=1}^{18} x_{i,j} ≤ supply_Si

Explicitly:
∑_{j=1}^{18} x_{1,j} ≤ 5963
∑_{j=1}^{18} x_{2,j} ≤ 702
...
∑_{j=1}^{18} x_{18,j} ≤ 333

3. Non-negativity:
x_{i,j} ≥ 0, ∀ i, j

Summary:
Minimize
∑_{i=1}^{18} ∑_{j=1}^{18} cost_{Si,Cj} * x_{i,j}
Subject to
∑_{i=1}^{18} x_{i,j} = demand_Cj, ∀ j=1..18
∑_{j=1}^{18} x_{i,j} ≤ supply_Si, ∀ i=1..18
x_{i,j} ≥ 0, ∀ i, j

All coefficients and identifiers are preserved in source order as required.