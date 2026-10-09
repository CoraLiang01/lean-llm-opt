Let:
- S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10} be the set of suppliers.
- C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10} be the set of customer groups.
- x_{i,j} = number of units transported from supplier i ∈ S to customer group j ∈ C.

Parameters:
- supply_capacity_i: daily supply capacity of supplier i.
- demand_j: daily demand of customer group j.
- cost_{i,j}: transportation cost per unit from supplier i to customer group j.

Data (preserving source order):

Supply capacities (from supply_capacity.csv):
- supply_capacity_S1 = 288
- supply_capacity_S2 = 288
- supply_capacity_S3 = 264
- supply_capacity_S4 = 264
- supply_capacity_S5 = 216
- supply_capacity_S6 = 216
- supply_capacity_S7 = 168
- supply_capacity_S8 = 216
- supply_capacity_S9 = 240
- supply_capacity_S10 = 168

Customer demands (from customer_demand.csv):
- demand_C1 = 216
- demand_C2 = 168
- demand_C3 = 264
- demand_C4 = 216
- demand_C5 = 216
- demand_C6 = 192
- demand_C7 = 144
- demand_C8 = 168
- demand_C9 = 168
- demand_C10 = 168

Transportation costs (from transportation_costs.csv):

|        |   C1   |   C2   |   C3   |   C4   |   C5   |   C6   |   C7   |   C8   |   C9   |   C10  |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| S1     | 590.3648136504455 | 23.669172607322494 | 88.89005869765714 | 497.52228807074613 | 466.09034321595647 | 29.022096827063212 | 23.675244833973835 | 23.677760288117437 | 0.3118394914937161 | 58.895473920714416 |
| S2     | 2042.0715001593626 | 2133.978484314172 | 705.15912033561 | 101.59454516295598 | 2052.937657376311 | 1738.754951414345 | 101.61094965654742 | 101.61062174376057 | 122.45214268700826 | 67.29170751036096 |
| S3     | 22.297222160217984 | 497.9271939314995 | 1653.0828862073263 | 23.68545123339267 | 1386.0807887282344 | 26.13715280763276 | 497.6220482906461 | 498.0935847133144 | 865.3816296318804 | 1008.6717394620979 |
| S4     | 960.7814533858373 | 49.128300053752405 | 1324.238697073691 | 1032.2095478151716 | 0.07804725392720868 | 53.308268726049285 | 49.1364167175093 | 1031.8214424894484 | 466.00495307991264 | 1351.8189071012546 |
| S5     | 1471.2721666392908 | 85.6956072820555 | 38.89266823851542 | 1542.0500358120464 | 112.20514372003504 | 82.3702016356405 | 1542.3399196620971 | 85.69238745806277 | 1924.9360769614245 | 1094.6695960752636 |
| S6     | 191.9058726130392 | 158.50401031820448 | 91.02045349777458 | 184.44747201726193 | 968.146798696633 | 284.1076062070199 | 8.791061587686942 | 158.70523835548545 | 27.943874345249665 | 929.807168280051 |
| S7     | 81.23891457326876 | 0.3744642223062507 | 2079.46686537067 | 0.3065671755503025 | 1031.7772962191823 | 7.203964492497209 | 0.07623072241762692 | 0.032473879548006554 | 23.685827966421357 | 849.9799406578097 |
| S8     | 56.099310965461356 | 935.6143108671334 | 73.08824617002863 | 52.00392409272077 | 4.025792388934198 | 1002.2327657984296 | 935.7766029588662 | 935.7007252277288 | 612.8698719325438 | 1348.8366145919845 |
| S9     | 4.502283326860296 | 0.3899585342810754 | 1782.4662178163346 | 0.006345906612718274 | 1031.9910114913148 | 129.50665619510303 | 0.2118319573481142 | 0.645730107353115 | 497.62723911435927 | 40.46575554562011 |
| S10    | 333.6869270439132 | 277.4719386113677 | 86.02096892455509 | 277.30836609256806 | 1004.4649084520337 | 19.950336815857597 | 13.202073690286834 | 238.14321521805866 | 411.0580332361589 | 941.7526365563969 |

Mathematical Model:

Variables:
- x_{i,j} ≥ 0, ∀ i ∈ S, j ∈ C

Objective:
Minimize total transportation cost:
\[
\min \sum_{i \in S} \sum_{j \in C} cost_{i,j} \cdot x_{i,j}
\]
That is,
\[
\min \Bigg(
590.3648136504455\,x_{S1,C1} + 23.669172607322494\,x_{S1,C2} + \ldots + 941.7526365563969\,x_{S10,C10}
\Bigg)
\]
(with all coefficients as above, in source order).

Subject to:

1. Supply capacity constraints (for each supplier i):
\[
\sum_{j \in C} x_{i,j} \leq supply\_capacity_i \quad \forall i \in S
\]
Explicitly:
\[
x_{S1,C1} + x_{S1,C2} + \ldots + x_{S1,C10} \leq 288
\]
\[
x_{S2,C1} + x_{S2,C2} + \ldots + x_{S2,C10} \leq 288
\]
...
\[
x_{S10,C1} + x_{S10,C2} + \ldots + x_{S10,C10} \leq 168
\]

2. Demand satisfaction constraints (for each customer group j):
\[
\sum_{i \in S} x_{i,j} = demand_j \quad \forall j \in C
\]
Explicitly:
\[
x_{S1,C1} + x_{S2,C1} + \ldots + x_{S10,C1} = 216
\]
\[
x_{S1,C2} + x_{S2,C2} + \ldots + x_{S10,C2} = 168
\]
...
\[
x_{S1,C10} + x_{S2,C10} + \ldots + x_{S10,C10} = 168
\]

3. Non-negativity:
\[
x_{i,j} \geq 0 \quad \forall i \in S, j \in C
\]

This is a complete numerical linear programming formulation of the transportation problem as described, using all identifiers and coefficients from the provided CSVs, preserving source order.