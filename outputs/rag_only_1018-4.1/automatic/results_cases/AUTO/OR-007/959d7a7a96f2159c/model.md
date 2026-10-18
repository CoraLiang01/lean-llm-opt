Let:
- S = {S1, S2, S3, S4, S5} be the set of warehouses (regions).
- D = {D1, D2, D3, D4, D5} be the set of stores (customers).
- x_{s,d} = quantity shipped from warehouse s ∈ S to store d ∈ D (decision variables, x_{s,d} ≥ 0).

Parameters (from CSVs, in source order):

Store demands (customer_demand.csv):
- demand_D1 = 428
- demand_D2 = 217
- demand_D3 = 214
- demand_D4 = 380
- demand_D5 = 254

Warehouse supply capacities (supply_capacity.csv):
- supply_S1 = 428
- supply_S2 = 217
- supply_S3 = 214
- supply_S4 = 380
- supply_S5 = 254

Transportation costs (transportation_costs.csv):

|         | D1              | D2              | D3              | D4              | D5              |
|---------|-----------------|-----------------|-----------------|-----------------|-----------------|
| S1      | 269.39105880208 | 1.45373353909   | 99.60345345757  | 26.64078166310  | 9.53768895688   |
| S2      | 9.29184687679   | 10.87477843707  | 144.52609291615 | 11.42013307790  | 153.17568199278 |
| S3      | 9.67458430167   | 2.61916509597   | 100.82422491687 | 3.21219108879   | 133.84933961242 |
| S4      | 270.57498480010 | 32.50253586     | 4.68420980965   | 1.56822696865   | 9.58927599      |
| S5      | 226.03319106758 | 8.66916198083   | 65.47681316968  | 9.06876525846   | 202.65015316426 |

Objective:
Minimize total transportation cost:
\[
\min \sum_{s \in S} \sum_{d \in D} c_{s,d} \cdot x_{s,d}
\]
where c_{s,d} are the costs from the table above, in source order.

Explicitly:
\[
\min \Bigg[
269.3910588020795\,x_{S1,D1} + 1.4537335390933939\,x_{S1,D2} + 99.60345345756605\,x_{S1,D3} + 26.64078166309837\,x_{S1,D4} + 9.537688956880922\,x_{S1,D5}
\]
\[
+ 9.291846876785183\,x_{S2,D1} + 10.874778437070223\,x_{S2,D2} + 144.52609291614627\,x_{S2,D3} + 11.420133077898234\,x_{S2,D4} + 153.1756819927813\,x_{S2,D5}
\]
\[
+ 9.674584301671008\,x_{S3,D1} + 2.6191650959687944\,x_{S3,D2} + 100.8242249168735\,x_{S3,D3} + 3.2121910887916876\,x_{S3,D4} + 133.8493396124168\,x_{S3,D5}
\]
\[
+ 270.57498480010247\,x_{S4,D1} + 32.50253586\,x_{S4,D2} + 4.6842098096469815\,x_{S4,D3} + 1.5682269686546804\,x_{S4,D4} + 9.58927599\,x_{S4,D5}
\]
\[
+ 226.0331910675782\,x_{S5,D1} + 8.669161980826471\,x_{S5,D2} + 65.47681316968448\,x_{S5,D3} + 9.068765258459958\,x_{S5,D4} + 202.65015316425533\,x_{S5,D5}
\Bigg]
\]

Subject to:

1. Demand satisfaction for each store (from customer_demand.csv, in source order):
\[
x_{S1,D1} + x_{S2,D1} + x_{S3,D1} + x_{S4,D1} + x_{S5,D1} = 428
\]
\[
x_{S1,D2} + x_{S2,D2} + x_{S3,D2} + x_{S4,D2} + x_{S5,D2} = 217
\]
\[
x_{S1,D3} + x_{S2,D3} + x_{S3,D3} + x_{S4,D3} + x_{S5,D3} = 214
\]
\[
x_{S1,D4} + x_{S2,D4} + x_{S3,D4} + x_{S4,D4} + x_{S5,D4} = 380
\]
\[
x_{S1,D5} + x_{S2,D5} + x_{S3,D5} + x_{S4,D5} + x_{S5,D5} = 254
\]

2. Supply capacity for each warehouse (from supply_capacity.csv, in source order):
\[
x_{S1,D1} + x_{S1,D2} + x_{S1,D3} + x_{S1,D4} + x_{S1,D5} \leq 428
\]
\[
x_{S2,D1} + x_{S2,D2} + x_{S2,D3} + x_{S2,D4} + x_{S2,D5} \leq 217
\]
\[
x_{S3,D1} + x_{S3,D2} + x_{S3,D3} + x_{S3,D4} + x_{S3,D5} \leq 214
\]
\[
x_{S4,D1} + x_{S4,D2} + x_{S4,D3} + x_{S4,D4} + x_{S4,D5} \leq 380
\]
\[
x_{S5,D1} + x_{S5,D2} + x_{S5,D3} + x_{S5,D4} + x_{S5,D5} \leq 254
\]

3. Nonnegativity:
\[
x_{s,d} \geq 0 \quad \forall s \in S, d \in D
\]

All identifiers, coefficients, and constraints are preserved in source order and as specified in the original query.