Let x_{ij} denote the number of units shipped from warehouse Si to store Dj, where i ∈ {1,2,3,4,5} and j ∈ {1,2,3,4,5}.

Parameters (from CSVs, source order preserved):

Store Demands (customer_demand.csv):
- D1: 428
- D2: 217
- D3: 214
- D4: 380
- D5: 254

Warehouse Supply Capacities (supply_capacity.csv):
- S1: 428
- S2: 217
- S3: 214
- S4: 380
- S5: 254

Transportation Costs per Unit (transportation_costs.csv):

|      | D1              | D2              | D3              | D4              | D5              |
|------|-----------------|-----------------|-----------------|-----------------|-----------------|
| S1   | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
| S2   | 9.29184687679   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
| S3   | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
| S4   | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
| S5   | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

Decision Variables:
x_{ij} ≥ 0, continuous, for all i ∈ {1,2,3,4,5}, j ∈ {1,2,3,4,5}

Objective (minimize total transportation cost, preserving all coefficients and structure):

Minimize
269.3910588020795 x_{1,1} + 1.4537335390933939 x_{1,2} + 99.60345345756605 x_{1,3} + 26.64078166309837 x_{1,4} + 9.537688956880922 x_{1,5}
+ 9.291846876785183 x_{2,1} + 10.874778437070223 x_{2,2} + 144.52609291614627 x_{2,3} + 11.420133077898234 x_{2,4} + 153.1756819927813 x_{2,5}
+ 9.674584301671008 x_{3,1} + 2.6191650959687944 x_{3,2} + 100.8242249168735 x_{3,3} + 3.2121910887916876 x_{3,4} + 133.8493396124168 x_{3,5}
+ 270.57498480010247 x_{4,1} + 32.50253586 x_{4,2} + 4.6842098096469815 x_{4,3} + 1.5682269686546804 x_{4,4} + 9.58927599 x_{4,5}
+ 226.0331910675782 x_{5,1} + 8.669161980826471 x_{5,2} + 65.47681316968448 x_{5,3} + 9.068765258459958 x_{5,4} + 202.65015316425533 x_{5,5}

Subject to:

Demand satisfaction for each store (from customer_demand.csv, source order):
x_{1,1} + x_{2,1} + x_{3,1} + x_{4,1} + x_{5,1} = 428
x_{1,2} + x_{2,2} + x_{3,2} + x_{4,2} + x_{5,2} = 217
x_{1,3} + x_{2,3} + x_{3,3} + x_{4,3} + x_{5,3} = 214
x_{1,4} + x_{2,4} + x_{3,4} + x_{4,4} + x_{5,4} = 380
x_{1,5} + x_{2,5} + x_{3,5} + x_{4,5} + x_{5,5} = 254

Supply capacity for each warehouse (from supply_capacity.csv, source order):
x_{1,1} + x_{1,2} + x_{1,3} + x_{1,4} + x_{1,5} ≤ 428
x_{2,1} + x_{2,2} + x_{2,3} + x_{2,4} + x_{2,5} ≤ 217
x_{3,1} + x_{3,2} + x_{3,3} + x_{3,4} + x_{3,5} ≤ 214
x_{4,1} + x_{4,2} + x_{4,3} + x_{4,4} + x_{4,5} ≤ 380
x_{5,1} + x_{5,2} + x_{5,3} + x_{5,4} + x_{5,5} ≤ 254

Variable domains:
x_{ij} ≥ 0 for all i ∈ {1,2,3,4,5}, j ∈ {1,2,3,4,5}

This is a complete numerical formulation of the GreenMart transportation problem, preserving all source order and coefficients.