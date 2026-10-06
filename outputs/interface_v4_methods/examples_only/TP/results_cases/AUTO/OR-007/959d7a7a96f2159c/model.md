Let:
- S = {S1, S2, S3, S4, S5} be the set of warehouses (sources)
- D = {D1, D2, D3, D4, D5} be the set of stores (destinations)
- x_{ij} = quantity shipped from warehouse i ∈ S to store j ∈ D (decision variables, x_{ij} ≥ 0)

Parameters (from CSVs):

Store Demands (customer_demand.csv):
- demand_D1 = 428
- demand_D2 = 217
- demand_D3 = 214
- demand_D4 = 380
- demand_D5 = 254

Warehouse Supply Capacities (supply_capacity.csv):
- supply_S1 = 428
- supply_S2 = 217
- supply_S3 = 214
- supply_S4 = 380
- supply_S5 = 254

Transportation Costs (transportation_costs.csv):

|        | D1           | D2           | D3           | D4           | D5           |
|--------|--------------|--------------|--------------|--------------|--------------|
| S1     | 269.3910588  | 1.453733539  | 99.60345346  | 26.64078166  | 9.537688957  |
| S2     | 9.291846877  | 10.87477844  | 144.5260929  | 11.42013308  | 153.17568199 |
| S3     | 9.674584302  | 2.619165096  | 100.8242249  | 3.212191089  | 133.84933961 |
| S4     | 270.5749848  | 32.50253586  | 4.68420981   | 1.568226969  | 9.58927599   |
| S5     | 226.0331911  | 8.669161981  | 65.47681317  | 9.068765258  | 202.65015316 |

Model:

Variables:
x_{ij} ≥ 0, ∀ i ∈ S, j ∈ D

Objective:
Minimize total transportation cost:
minimize
269.3910588020795 x_{S1,D1} + 1.4537335390933939 x_{S1,D2} + 99.60345345756605 x_{S1,D3} + 26.64078166309837 x_{S1,D4} + 9.537688956880922 x_{S1,D5}
+ 9.291846876785183 x_{S2,D1} + 10.874778437070223 x_{S2,D2} + 144.52609291614627 x_{S2,D3} + 11.420133077898234 x_{S2,D4} + 153.1756819927813 x_{S2,D5}
+ 9.674584301671008 x_{S3,D1} + 2.6191650959687944 x_{S3,D2} + 100.8242249168735 x_{S3,D3} + 3.2121910887916876 x_{S3,D4} + 133.8493396124168 x_{S3,D5}
+ 270.57498480010247 x_{S4,D1} + 32.50253586 x_{S4,D2} + 4.6842098096469815 x_{S4,D3} + 1.5682269686546804 x_{S4,D4} + 9.58927599 x_{S4,D5}
+ 226.0331910675782 x_{S5,D1} + 8.669161980826471 x_{S5,D2} + 65.47681316968448 x_{S5,D3} + 9.068765258459958 x_{S5,D4} + 202.65015316425533 x_{S5,D5}

Subject to:

Demand satisfaction (for each store):
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} = 428
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} = 217
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} = 214
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} = 380
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} = 254

Supply capacity (for each warehouse):
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} ≤ 428
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} ≤ 217
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} ≤ 214
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} ≤ 380
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} ≤ 254

Non-negativity:
x_{ij} ≥ 0, ∀ i ∈ S, j ∈ D

This is the complete numerical formulation of the GreenMart transportation optimization problem, preserving all source-ordered identifiers and coefficients.