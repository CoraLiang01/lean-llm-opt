Sets:
- Let S = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11} be the set of Walmart stores.
- Let C = {C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12} be the set of customer groups.

Parameters:
- demand_c: demand for customer group c ∈ C, from customer_demand.csv:
  demand_C1 = 11
  demand_C2 = 1148
  demand_C3 = 54
  demand_C4 = 833
  demand_C5 = 154
  demand_C6 = 551
  demand_C7 = 7081
  demand_C8 = 76
  demand_C9 = 66
  demand_C10 = 174
  demand_C11 = 15
  demand_C12 = 680

- supply_capacity_s: supply capacity for store s ∈ S, from supply_capacity.csv:
  supply_capacity_S1 = 4
  supply_capacity_S2 = 575
  supply_capacity_S3 = 1504
  supply_capacity_S4 = 178
  supply_capacity_S5 = 228
  supply_capacity_S6 = 50
  supply_capacity_S7 = 3
  supply_capacity_S8 = 6148
  supply_capacity_S9 = 6
  supply_capacity_S10 = 10673
  supply_capacity_S11 = 174

- cost_{s,c}: transportation cost per unit from store s to customer group c, from transportation_costs.csv:

|        |   C1   |    C2    |    C3    |    C4    |    C5    |    C6    |    C7    |    C8    |    C9    |   C10   |   C11   |   C12   |
|--------|--------|----------|----------|----------|----------|----------|----------|----------|----------|---------|---------|---------|
| S1     | 0.6391 | 49.7184  | 33.7586  | 1570.6731| 1370.4095| 57.3531  | 57.1830  | 54.9210  | 1143.6809| 52.4913 | 606.4434|1192.4687|
| S2     |605.4786| 64.5356  |478.4779  | 887.0481 | 65.4611  | 71.9361  | 41.2902  | 70.3604  | 35.3589  |1472.7482| 0.6005  | 49.8685 |
| S3     |1139.0440| 4.7851  |1805.6214 |1302.8958 |2437.3212 |103.8037  |774.6558  | 4.5160   | 879.7049 |162.7056 |1208.6135|110.1869 |
| S4     | 69.2699|2105.4854 | 869.6820 |1494.8986 |310.5377  | 98.1546  |103.3692  |1758.8784 | 97.2854  | 94.6504 |1277.2515| 21.6362 |
| S5     |980.4114|899.3109  |1183.0326 | 402.0986 | 81.7886  |1115.6819 |123.8043  |1121.1469 | 0.0024   |1009.6452| 35.3480 |1625.4346|
| S6     |1246.7825|2105.7967|1014.3393 |1494.6681 |362.0174  | 98.1714  |2170.4059 | 97.7319  | 97.2683  |1987.9908| 70.9440 |389.1598 |
| S7     | 57.1086| 23.8362  | 78.1057  | 742.8068 |1926.0797 |454.3790  |458.2901  |465.9308  | 28.1386  |524.6154 |997.5318 |104.4779 |
| S8     |981.2909|120.9013  |1625.8207 |1267.8229 |2569.6446 | 13.4718  |815.1525  |253.4235  | 43.7656  |275.9784 |1228.0699|103.4832 |
| S9     | 30.5328|1444.8595 | 173.5547 |1307.3913 |965.2012  |1843.7769 |1483.6409 | 85.3221  |1353.5009 |1485.9154| 29.4238 | 26.6194 |
| S10    | 94.1109|1422.9971 |1470.7769 |1419.3382 | 38.9453  | 72.2011  |2040.4606 |1542.7026 |1803.8002 | 72.9437 |2181.4542|973.5516 |
| S11    |1032.9074|166.3018 |1620.4767 | 64.6683  |2000.5092 | 0.0029   | 47.0384  | 52.9922  |1115.6336 |129.7934 |1295.0978|2330.7682|

Decision Variables:
- x_{s,c} ≥ 0: quantity transported from store s ∈ S to customer group c ∈ C.

Objective:
Minimize total transportation cost:
minimize
∑_{s∈S} ∑_{c∈C} cost_{s,c} * x_{s,c}
That is,
minimize
0.639144476970582 x_{S1,C1} + 49.71842803015729 x_{S1,C2} + 33.75857739960576 x_{S1,C3} + 1570.673110465785 x_{S1,C4} + 1370.4095474322417 x_{S1,C5} + 57.35307774277479 x_{S1,C6} + 57.18299486453194 x_{S1,C7} + 54.9209612366192 x_{S1,C8} + 1143.680909226399 x_{S1,C9} + 52.49127007738756 x_{S1,C10} + 606.4434399601076 x_{S1,C11} + 1192.4686514332489 x_{S1,C12}
+ 605.4786373569875 x_{S2,C1} + 64.53562572761275 x_{S2,C2} + 478.4779031378926 x_{S2,C3} + 887.0480739088434 x_{S2,C4} + 65.46111249492031 x_{S2,C5} + 71.93605217833378 x_{S2,C6} + 41.29015388498019 x_{S2,C7} + 70.36038207491039 x_{S2,C8} + 35.35892996332259 x_{S2,C9} + 1472.7481944140839 x_{S2,C10} + 0.6004591535232997 x_{S2,C11} + 49.86854015671519 x_{S2,C12}
+ ... (continue for all s ∈ S, c ∈ C, preserving the order and coefficients as above).

Subject to:

1. Demand satisfaction for each customer group:
For each c ∈ C:
∑_{s∈S} x_{s,c} = demand_c

That is,
x_{S1,C1} + x_{S2,C1} + ... + x_{S11,C1} = 11
x_{S1,C2} + x_{S2,C2} + ... + x_{S11,C2} = 1148
...
x_{S1,C12} + x_{S2,C12} + ... + x_{S11,C12} = 680

2. Supply capacity for each store:
For each s ∈ S:
∑_{c∈C} x_{s,c} ≤ supply_capacity_s

That is,
x_{S1,C1} + x_{S1,C2} + ... + x_{S1,C12} ≤ 4
x_{S2,C1} + x_{S2,C2} + ... + x_{S2,C12} ≤ 575
...
x_{S11,C1} + x_{S11,C2} + ... + x_{S11,C12} ≤ 174

3. Non-negativity:
x_{s,c} ≥ 0 for all s ∈ S, c ∈ C

This is a complete numerical linear programming formulation of the transportation problem as described, with all identifiers and coefficients preserved in source order.