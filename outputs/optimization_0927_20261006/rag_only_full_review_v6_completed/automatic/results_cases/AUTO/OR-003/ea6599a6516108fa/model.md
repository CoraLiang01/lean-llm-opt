Sets:
- Let S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10} be the set of suppliers.
- Let C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10} be the set of customer groups.

Parameters:
- Supply capacities (units per day):
  - cap[S1] = 288
  - cap[S2] = 288
  - cap[S3] = 264
  - cap[S4] = 264
  - cap[S5] = 216
  - cap[S6] = 216
  - cap[S7] = 168
  - cap[S8] = 216
  - cap[S9] = 240
  - cap[S10] = 168

- Customer demands (units per day):
  - dem[C1] = 216
  - dem[C2] = 168
  - dem[C3] = 264
  - dem[C4] = 216
  - dem[C5] = 216
  - dem[C6] = 192
  - dem[C7] = 144
  - dem[C8] = 168
  - dem[C9] = 168
  - dem[C10] = 168

- Transportation costs per unit (cost[S, C]):
  - cost[S1, C1] = 590.3648136504455
  - cost[S1, C2] = 23.669172607322494
  - cost[S1, C3] = 88.89005869765714
  - cost[S1, C4] = 497.52228807074613
  - cost[S1, C5] = 466.09034321595647
  - cost[S1, C6] = 29.022096827063212
  - cost[S1, C7] = 23.675244833973835
  - cost[S1, C8] = 23.677760288117437
  - cost[S1, C9] = 0.3118394914937161
  - cost[S1, C10] = 58.895473920714416

  - cost[S2, C1] = 2042.0715001593626
  - cost[S2, C2] = 2133.978484314172
  - cost[S2, C3] = 705.15912033561
  - cost[S2, C4] = 101.59454516295598
  - cost[S2, C5] = 2052.937657376311
  - cost[S2, C6] = 1738.754951414345
  - cost[S2, C7] = 101.61094965654742
  - cost[S2, C8] = 101.61062174376057
  - cost[S2, C9] = 122.45214268700826
  - cost[S2, C10] = 67.29170751036096

  - cost[S3, C1] = 22.297222160217984
  - cost[S3, C2] = 497.9271939314995
  - cost[S3, C3] = 1653.0828862073263
  - cost[S3, C4] = 23.68545123339267
  - cost[S3, C5] = 1386.0807887282344
  - cost[S3, C6] = 26.13715280763276
  - cost[S3, C7] = 497.6220482906461
  - cost[S3, C8] = 498.0935847133144
  - cost[S3, C9] = 865.3816296318804
  - cost[S3, C10] = 1008.6717394620979

  - cost[S4, C1] = 960.7814533858373
  - cost[S4, C2] = 49.128300053752405
  - cost[S4, C3] = 1324.238697073691
  - cost[S4, C4] = 1032.2095478151716
  - cost[S4, C5] = 0.07804725392720868
  - cost[S4, C6] = 53.308268726049285
  - cost[S4, C7] = 49.1364167175093
  - cost[S4, C8] = 1031.8214424894484
  - cost[S4, C9] = 466.00495307991264
  - cost[S4, C10] = 1351.8189071012546

  - cost[S5, C1] = 1471.2721666392908
  - cost[S5, C2] = 85.6956072820555
  - cost[S5, C3] = 38.89266823851542
  - cost[S5, C4] = 1542.0500358120464
  - cost[S5, C5] = 112.20514372003504
  - cost[S5, C6] = 82.3702016356405
  - cost[S5, C7] = 1542.3399196620971
  - cost[S5, C8] = 85.69238745806277
  - cost[S5, C9] = 1924.9360769614245
  - cost[S5, C10] = 1094.6695960752636

  - cost[S6, C1] = 191.9058726130392
  - cost[S6, C2] = 158.50401031820448
  - cost[S6, C3] = 91.02045349777458
  - cost[S6, C4] = 184.44747201726193
  - cost[S6, C5] = 968.146798696633
  - cost[S6, C6] = 284.1076062070199
  - cost[S6, C7] = 8.791061587686942
  - cost[S6, C8] = 158.70523835548545
  - cost[S6, C9] = 27.943874345249665
  - cost[S6, C10] = 929.807168280051

  - cost[S7, C1] = 81.23891457326876
  - cost[S7, C2] = 0.3744642223062507
  - cost[S7, C3] = 2079.46686537067
  - cost[S7, C4] = 0.3065671755503025
  - cost[S7, C5] = 1031.7772962191823
  - cost[S7, C6] = 7.203964492497209
  - cost[S7, C7] = 0.07623072241762692
  - cost[S7, C8] = 0.032473879548006554
  - cost[S7, C9] = 23.685827966421357
  - cost[S7, C10] = 849.9799406578097

  - cost[S8, C1] = 56.099310965461356
  - cost[S8, C2] = 935.6143108671334
  - cost[S8, C3] = 73.08824617002863
  - cost[S8, C4] = 52.00392409272077
  - cost[S8, C5] = 4.025792388934198
  - cost[S8, C6] = 1002.2327657984296
  - cost[S8, C7] = 935.7766029588662
  - cost[S8, C8] = 935.7007252277288
  - cost[S8, C9] = 612.8698719325438
  - cost[S8, C10] = 1348.8366145919845

  - cost[S9, C1] = 4.502283326860296
  - cost[S9, C2] = 0.3899585342810754
  - cost[S9, C3] = 1782.4662178163346
  - cost[S9, C4] = 0.006345906612718274
  - cost[S9, C5] = 1031.9910114913148
  - cost[S9, C6] = 129.50665619510303
  - cost[S9, C7] = 0.2118319573481142
  - cost[S9, C8] = 0.645730107353115
  - cost[S9, C9] = 497.62723911435927
  - cost[S9, C10] = 40.46575554562011

  - cost[S10, C1] = 333.6869270439132
  - cost[S10, C2] = 277.4719386113677
  - cost[S10, C3] = 86.02096892455509
  - cost[S10, C4] = 277.30836609256806
  - cost[S10, C5] = 1004.4649084520337
  - cost[S10, C6] = 19.950336815857597
  - cost[S10, C7] = 13.202073690286834
  - cost[S10, C8] = 238.14321521805866
  - cost[S10, C9] = 411.0580332361589
  - cost[S10, C10] = 941.7526365563969

Decision Variables:
- Let x[S, C] ≥ 0: the number of units shipped from supplier S to customer group C.

Objective:
Minimize total transportation cost:
\[
\text{Minimize} \quad Z = \sum_{S \in S} \sum_{C \in C} \text{cost}[S, C] \cdot x[S, C]
\]

Subject to:

1. Supply capacity constraints (for each supplier S):
\[
\sum_{C \in C} x[S, C] \leq \text{cap}[S] \quad \forall S \in S
\]

2. Demand satisfaction constraints (for each customer group C):
\[
\sum_{S \in S} x[S, C] = \text{dem}[C] \quad \forall C \in C
\]

3. Non-negativity:
\[
x[S, C] \geq 0 \quad \forall S \in S, \forall C \in C
\]

This is a complete numerical linear programming formulation of the transportation problem as described, using all identifiers and coefficients from the provided CSVs, preserving source order and all original data.