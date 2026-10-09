Let S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10} be the set of warehouses, and C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10} be the set of customers (stores).

Decision Variables:
x_{i,j} = quantity of product shipped from warehouse i ∈ S to customer j ∈ C
Domain: x_{i,j} ≥ 0, continuous

Parameters:
Demands (from customer_demand.csv, in source order):
demand_C1 = 45
demand_C2 = 23
demand_C3 = 94
demand_C4 = 92
demand_C5 = 57
demand_C6 = 52
demand_C7 = 23
demand_C8 = 99
demand_C9 = 99
demand_C10 = 77

Supply capacities (from supply_capacity.csv, in source order):
supply_S1 = 127
supply_S2 = 236
supply_S3 = 168
supply_S4 = 115
supply_S5 = 280
supply_S6 = 179
supply_S7 = 135
supply_S8 = 263
supply_S9 = 283
supply_S10 = 476

Transportation costs (from transportation_costs.csv, in source order):

|         | C1           | C2           | C3           | C4           | C5           | C6           | C7           | C8           | C9           | C10          |
|---------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| S1      | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S2      | 2077.0586725 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 8.063346156  | 0.0          |
| S3      | 79.92102960  | 474.24509131 | 1477.0676289 | 22.58309959  | 474.24509131 | 41.10659696  | 474.24509131 | 474.24509131 | 624.16253950 | 474.24509131 |
| S4      | 1659.3369291 | 57.20541469  | 186.15190481 | 1201.3137084 | 1029.6974644 | 41.82210594  | 57.20541469  | 1201.3137084 | 884.56338707 | 1029.6974644 |
| S5      | 1297.2567041 | 77.76629131  | 24.26760228  | 1399.7932436 | 77.76629131  | 53.91161728  | 1399.7932436 | 77.76629131  | 1255.1151480 | 1399.7932436 |
| S6      | 1998.9090659 | 985.31654357 | 2.854168689  | 1149.5359675 | 985.31654357 | 730.69236477 | 54.73980798  | 985.31654357 | 46.80310221  | 1149.5359675 |
| S7      | 1780.3360050 | 0.0          | 1141.0405609 | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 8.063346156  | 0.0          |
| S8      | 75.40935896  | 1338.1987291 | 21.39134599  | 74.34437384  | 74.34437384  | 937.35062391 | 1338.1987291 | 1338.1987291 | 1392.1186581 | 1338.1987291 |
| S9      | 98.90755583  | 0.0          | 978.03476648 | 0.0          | 0.0          | 651.11123325 | 0.0          | 0.0          | 169.33026927 | 0.0          |
| S10     | 2077.0586725 | 0.0          | 54.33526480  | 0.0          | 0.0          | 36.17284629  | 0.0          | 0.0          | 145.14023080 | 0.0          |

Mathematical Formulation:

Minimize total transportation cost:
minimize
∑_{i∈S} ∑_{j∈C} cost_{i,j} * x_{i,j}
where cost_{i,j} is as given in the table above.

Subject to:

1. Demand satisfaction for each customer:
For each j ∈ C:
  ∑_{i∈S} x_{i,j} = demand_Cj

2. Supply capacity for each warehouse:
For each i ∈ S:
  ∑_{j∈C} x_{i,j} ≤ supply_Si

3. Non-negativity:
For all i ∈ S, j ∈ C:
  x_{i,j} ≥ 0

All coefficients and identifiers are preserved in source order. No data is omitted or aggregated.